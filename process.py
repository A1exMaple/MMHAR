import os
import glob
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict

# ------------------ 保存 PT 文件 ------------------
def save_separate_pt(data_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, sample in enumerate(data_list):
        base = f"sample_{idx}"
        torch.save(sample["rgb"],   os.path.join(save_dir, base + "_rgb.pt"))
        torch.save(sample["imu"],   os.path.join(save_dir, base + "_imu.pt"))
        torch.save(sample["label"], os.path.join(save_dir, base + "_label.pt"))

# ------------------ 视频读取 ------------------
def load_video_frames(video_path, resize=(112, 112), max_frames=16):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    frames = [frames[i] for i in idx]
    frames = np.stack(frames, axis=0)
    return frames, fps, len(frames)

# ------------------ RGB 归一化 ------------------
def frames_to_normalized_tensors(frames):
    video = frames.astype(np.float32) / 255.0
    video = torch.from_numpy(video.transpose(3, 0, 1, 2))
    mean = torch.tensor([0.43216,0.394666,0.37645]).view(3,1,1,1)
    std  = torch.tensor([0.22803,0.22145,0.216989]).view(3,1,1,1)
    return (video - mean) / std

# ------------------ IMU 增强（保留） ------------------
def enhance_with_diff(x):
    if x.shape[0] < 2:
        diff = torch.zeros_like(x)
    else:
        diff = torch.cat([torch.zeros_like(x[:1]), x[1:] - x[:-1]], dim=0)
    return torch.cat([x, diff], dim=1)  # [T, 2C]

# ------------------ IMU CSV 加载 ------------------
def load_imu_csv(path, dim):
    if path is None or not os.path.exists(path):
        print(f"⚠️ IMU file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"⚠️ IMU file is empty: {path}")
            return None
        return torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
    except pd.errors.EmptyDataError:
        print(f"⚠️ IMU file empty data error: {path}")
        return None

def interp_to_frames(data, T):
    """
    data: [T_src, C]
    return: [T, C]
    """
    if data.shape[0] == T:
        return data

    if data.shape[0] < 2:
        return data.repeat(T, 1)

    device = data.device
    T_src = data.shape[0]

    idx = torch.linspace(0, T_src - 1, T, device=device)
    idx0 = torch.floor(idx).long()
    idx1 = torch.clamp(idx0 + 1, max=T_src - 1)
    w = (idx - idx0).unsqueeze(1)

    return (1 - w) * data[idx0] + w * data[idx1]


# ------------------ IMU 加载（❗核心修改） ------------------
def load_aligned_imu(sensor_root, subject, scene, session, action):
    def p(mod):
        return os.path.join(sensor_root, mod, subject, scene, session, action + ".csv")

    acc_path  = p("acc_phone_clip")
    gyro_path = p("gyro_clip")
    ori_path  = p("orientation_clip")

    for path in [acc_path, gyro_path, ori_path]:
        if not os.path.exists(path):
            print(f"⚠️ 缺失 IMU 文件，跳过样本: {path}")
            return None

    acc  = load_imu_csv(acc_path, 3)
    gyro = load_imu_csv(gyro_path, 3)
    ori  = load_imu_csv(ori_path, 3)

    if acc is None or gyro is None or ori is None:
        return None

    # === 差分增强（你原来的逻辑，保留）===
    acc  = enhance_with_diff(acc)   # [T1, 6]
    gyro = enhance_with_diff(gyro)  # [T2, 6]
    ori  = enhance_with_diff(ori)   # [T3, 6]

    # === 关键修复：对齐到同一长度 ===
    T = max(acc.shape[0], gyro.shape[0], ori.shape[0])

    acc  = interp_to_frames(acc,  T)
    gyro = interp_to_frames(gyro, T)
    ori  = interp_to_frames(ori,  T)

    # === 现在一定可以 cat ===
    imu = torch.cat([acc, gyro, ori], dim=1)  # [T, 18]
    return imu


# ------------------ 路径解析 ------------------
def parse_video_path(path):
    parts = path.replace("\\", "/").split("/")
    subject = [p for p in parts if p.startswith("subject")][0]
    scene   = [p for p in parts if p.startswith("scene")][0]
    session = [p for p in parts if p.startswith("session")][0]
    action  = os.path.splitext(os.path.basename(path))[0]
    return subject, scene, session, action

# ------------------ 主预处理 ------------------
def preprocess_dataset(video_root, sensor_root, save_root, max_frames=16, cam="cam1"):
    action2label = {
        "carrying":0,"checking_time":1,"closing":2,"crouching":3,"entering":4,"exiting":5,
        "fall":6,"jumping":7,"kicking":8,"loitering":9,"looking_around":10,"opening":11,
        "picking_up":12,"pointing":13,"pulling":14,"pushing":15,"running":16,"setting_down":17,
        "standing":18,"talking":19,"talking_on_phone":20,"throwing":21,
        "transferring_object":22,"using_phone":23,"walking":24,"waving_hand":25,
        "drinking":26,"pocket_in":27,"pocket_out":28,"sitting":29,"sitting_down":30,
        "standing_up":31,"using_pc":32,"carrying_heavy":33,"carrying_light":34
    }

    videos = glob.glob(os.path.join(video_root, "subject*", cam, "scene*", "session*", "*.mp4"))

    subject2videos = defaultdict(list)
    for v in videos:
        subject, *_ = parse_video_path(v)
        subject2videos[subject].append(v)

    subjects = sorted(subject2videos.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)

    n = len(subjects)
    splits = {
        "train": sum([subject2videos[s] for s in subjects[:int(0.6*n)]], []),
        "val":   sum([subject2videos[s] for s in subjects[int(0.6*n):int(0.8*n)]], []),
        "test":  sum([subject2videos[s] for s in subjects[int(0.8*n):]], [])
    }

    for split, vids in splits.items():
        data = []
        print(f"\n🚀 Processing {split} ({len(vids)})")
        for v in tqdm(vids):
            subject, scene, session, action = parse_video_path(v)
            if action not in action2label:
                continue

            try:
                frames, _, _ = load_video_frames(v, max_frames=max_frames)
            except RuntimeError:
                continue

            rgb = frames_to_normalized_tensors(frames)
            imu = load_aligned_imu(sensor_root, subject, scene, session, action)
            if imu is None:
                continue

            label = torch.tensor(action2label[action], dtype=torch.long)
            data.append({"rgb": rgb, "imu": imu, "label": label})

        save_separate_pt(data, os.path.join(save_root, split))
        print(f"✅ {split} saved: {len(data)} samples")

# ------------------ main ------------------
if __name__ == "__main__":
    preprocess_dataset(
        video_root=r"D:\HAR\MMAct\trimmed",
        sensor_root=r"D:\HAR\MMAct\trimmed_acc",
        save_root=r"D:\HAR\MMAct_preprocessed_proposed",
        max_frames=16,
        cam="cam1"
    )

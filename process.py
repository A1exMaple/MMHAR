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

# ------------------ IMU 增强（保持原逻辑） ------------------
def enhance_with_diff(x):
    if x.shape[0] < 2:
        diff = torch.zeros_like(x)
    else:
        diff = torch.cat([torch.zeros_like(x[:1]), x[1:] - x[:-1]], dim=0)
    return torch.cat([x, diff], dim=1)

# ------------------ IMU 精确加载 ------------------
def load_imu_csv(path, dim):
    if path is None or not os.path.exists(path):
        print(f"⚠️ IMU file not found: {path}")
        return torch.zeros((1, dim), dtype=torch.float32)
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"⚠️ IMU file is empty: {path}")
            return torch.zeros((1, dim), dtype=torch.float32)
        return torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
    except pd.errors.EmptyDataError:
        print(f"⚠️ IMU file empty data error: {path}")
        return torch.zeros((1, dim), dtype=torch.float32)

def interp_to_frames(data, T):
    if data.shape[0] == 1:
        return data.repeat(T, 1)
    idx = np.linspace(0, data.shape[0] - 1, T)
    idx0 = np.floor(idx).astype(int)
    idx1 = np.clip(idx0 + 1, 0, data.shape[0] - 1)
    w = idx - idx0
    return (1 - torch.tensor(w).unsqueeze(1)) * data[idx0] + torch.tensor(w).unsqueeze(1) * data[idx1]

# ------------------ IMU 对齐 ------------------
def load_aligned_imu(sensor_root, subject, scene, session, action, max_frames):
    def p(mod):
        return os.path.join(sensor_root, mod, subject, scene, session, action + ".csv")

    acc_path = p("acc_phone_clip")
    gyro_path = p("gyro_clip")
    ori_path = p("orientation_clip")

    # 检查 IMU 文件是否全部存在
    for path in [acc_path, gyro_path, ori_path]:
        if not os.path.exists(path):
            print(f"⚠️ 缺失 IMU 文件，跳过样本: {path}")
            return None  # 返回 None 表示这个样本缺失 IMU

    acc = enhance_with_diff(load_imu_csv(acc_path, 3))
    gyro = enhance_with_diff(load_imu_csv(gyro_path, 3))
    ori = enhance_with_diff(load_imu_csv(ori_path, 3))

    acc = interp_to_frames(acc, max_frames)
    gyro = interp_to_frames(gyro, max_frames)
    ori = interp_to_frames(ori, max_frames)

    return torch.cat([acc, gyro, ori], dim=1)


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
        "drinking":26, "pocket_in":27, "pocket_out":28, "sitting":29, "sitting_down":30,
        "standing_up":31,"using_pc":32,"carrying_heavy":33,"carrying_light":34
    }

    videos = glob.glob(os.path.join(video_root, "subject*", cam, "scene*", "session*", "*.mp4"))

    # cross-subject split
    subject2videos = defaultdict(list)
    for v in videos:
        subject, *_ = parse_video_path(v)
        subject2videos[subject].append(v)

    subjects = sorted(subject2videos.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)

    n = len(subjects)
    train_s = subjects[:int(0.6*n)]
    val_s   = subjects[int(0.6*n):int(0.8*n)]
    test_s  = subjects[int(0.8*n):]

    splits = {
        "train": sum([subject2videos[s] for s in train_s], []),
        "val":   sum([subject2videos[s] for s in val_s], []),
        "test":  sum([subject2videos[s] for s in test_s], [])
    }

    # ------------------ 核心循环：处理每个样本 ------------------
    for split, vids in splits.items():
        data = []
        print(f"\n🚀 Processing {split} ({len(vids)})")
        for v in tqdm(vids):
            subject, scene, session, action = parse_video_path(v)

            # 跳过未知动作
            if action not in action2label:
                print(f"⚠️ Skip unknown action: {action}")
                continue

            # 检查 IMU 文件是否存在
            imu_files_exist = all(os.path.exists(os.path.join(sensor_root, mod, subject, scene, session, action + ".csv"))
                                  for mod in ["acc_phone_clip", "gyro_clip", "orientation_clip"])
            if not imu_files_exist:
                print(f"⚠️ Skip sample with missing IMU: {subject}/{scene}/{session}/{action}")
                continue

            # 读取视频
            try:
                frames, fps, _ = load_video_frames(v, max_frames=max_frames)
            except RuntimeError:
                print(f"⚠️ Skip empty video: {v}")
                continue

            rgb = frames_to_normalized_tensors(frames)
            imu = load_aligned_imu(sensor_root, subject, scene, session, action, max_frames)
            if imu is None:
                # IMU 文件缺失，跳过这个样本
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
        save_root=r"D:\HAR\MMAct_preprocessed_clean",
        max_frames=16,
        cam="cam1"
    )

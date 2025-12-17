import os
import glob
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ------------------ 保存 PT 文件 ------------------
def save_separate_pt(data_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, sample in enumerate(data_list):
        base_name = f"sample_{idx}"
        torch.save(sample['rgb'], os.path.join(save_dir, base_name + "_rgb.pt"))
        torch.save(sample['imu'], os.path.join(save_dir, base_name + "_imu.pt"))
        torch.save(sample['label'], os.path.join(save_dir, base_name + "_label.pt"))

# ------------------ 视频读取（返回 uint8 frames） ------------------
def load_video_frames(video_path, resize=(112, 112), max_frames=16):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    video_len = len(frames)
    if video_len == 0:
        raise ValueError(f"无法读取视频帧: {video_path}")

    if video_len < max_frames:
        frames += [frames[-1].copy()] * (max_frames - video_len)
    else:
        idx = np.linspace(0, video_len - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    frames = np.stack(frames, axis=0)
    return frames, fps, frames.shape[0]

# ------------------ 归一化为 tensor ------------------
def frames_to_normalized_tensors(frames):
    video = frames.astype(np.float32)/255.0
    video_tensor = torch.from_numpy(video.transpose(3,0,1,2)).float()
    mean = torch.tensor([0.43216,0.394666,0.37645]).view(3,1,1,1)
    std = torch.tensor([0.22803,0.22145,0.216989]).view(3,1,1,1)
    video_tensor = (video_tensor - mean)/std
    return video_tensor

# ------------------ IMU 对齐 ------------------
def align_sensor_to_video(sensor_df, video_len, fps, max_frames=16):
    """
    改进版滑动窗口对 IMU 序列做加权池化
    输出 [T, C]
    """
    imu_data = sensor_df.iloc[:, 1:].values.astype(np.float32)  # [N, C]
    N, C = imu_data.shape

    if N == 0:
        return torch.zeros((max_frames, C), dtype=torch.float32)

    T = max_frames
    win_len = max(1, N // T)
    stride = max(1, win_len // 2)

    pooled = []
    start = 0

    while len(pooled) < T and start < N:
        end = min(start + win_len, N)
        seg = imu_data[start:end]

        if len(seg) == 0:
            pooled.append(imu_data[min(start, N - 1)])
        else:
            # ===== 加权池化，权重 = 每行 norm =====
            weights = np.linalg.norm(seg, axis=1, keepdims=True)
            weighted_mean = (seg * weights).sum(axis=0) / (weights.sum() + 1e-6)
            pooled.append(weighted_mean)

        start += stride

    aligned = np.stack(pooled, axis=0)

    # pad / truncate
    if aligned.shape[0] < max_frames:
        pad = np.tile(aligned[-1:], (max_frames - aligned.shape[0], 1))
        aligned = np.vstack([aligned, pad])
    else:
        aligned = aligned[:max_frames]

    return torch.tensor(aligned, dtype=torch.float32)


# ------------------ 模态增强 ------------------
def enhance_acc(acc):
    """
    acc: [T, 3] tensor
    输出: [T, 6] tensor, 包含原始加速度和差分特征
    """
    T = acc.shape[0]
    if T < 2:
        # 如果时间长度太短，diff 用全 0
        diff = torch.zeros_like(acc)
    else:
        diff = acc[1:] - acc[:-1]
        # 用 0 补齐第一行
        diff = torch.cat([torch.zeros((1, acc.shape[1]), device=acc.device), diff], dim=0)

    # 最终确保 diff 长度和 acc 一致
    if diff.shape[0] != T:
        pad_len = T - diff.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, diff.shape[1]), device=acc.device)
            diff = torch.cat([diff, pad], dim=0)
        elif pad_len < 0:
            diff = diff[:T]

    return torch.cat([acc, diff], dim=1)  # [T, 6]


def enhance_gyro(gyro):
    """
    gyro: [T, 3] tensor
    输出: [T, 6] tensor, 包含原始角速度和差分特征
    """
    T = gyro.shape[0]
    if T < 2:
        diff = torch.zeros_like(gyro)
    else:
        diff = gyro[1:] - gyro[:-1]
        diff = torch.cat([torch.zeros((1, gyro.shape[1]), device=gyro.device), diff], dim=0)

    if diff.shape[0] != T:
        pad_len = T - diff.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, diff.shape[1]), device=gyro.device)
            diff = torch.cat([diff, pad], dim=0)
        elif pad_len < 0:
            diff = diff[:T]

    return torch.cat([gyro, diff], dim=1)  # [T, 6]


def enhance_ori(ori):
    """
    ori: [T, 3] 或 [T, 4] tensor
    输出: [T, ori_dim*2], 包含原始 orientation + 差分
    """
    T = ori.shape[0]
    if T < 2:
        diff = torch.zeros_like(ori)
    else:
        diff = ori[1:] - ori[:-1]
        diff = torch.cat([torch.zeros((1, ori.shape[1]), device=ori.device), diff], dim=0)

    # 对齐长度
    if diff.shape[0] != T:
        pad_len = T - diff.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, diff.shape[1]), device=ori.device)
            diff = torch.cat([diff, pad], dim=0)
        elif pad_len < 0:
            diff = diff[:T]

    return torch.cat([ori, diff], dim=1)  # [T, ori_dim*2]

# ------------------ 对齐三个模态（安全版本） ------------------
def align_imu_modalities(acc_path, gyro_path, ori_path, video_len, fps, max_frames=16):
    """
    读取 acc / gyro / ori CSV，并做物理语义增强，保证长度一致。
    返回: [max_frames, total_dim] tensor
    """
    def load_csv_safe(path, default_dim=3):
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path)
            # print("源文件：", df.shape)
            data = torch.tensor(df.iloc[:,1:].values.astype(np.float32))
            if data.shape[0] == 0:
                data = torch.zeros((1, df.shape[1]-1), dtype=torch.float32)
        else:
            # print("没找到路径")
            data = torch.zeros((1, default_dim), dtype=torch.float32)
        return data

    # -------- 读取数据 --------
    acc = load_csv_safe(acc_path, default_dim=3)
    gyro = load_csv_safe(gyro_path, default_dim=3)
    ori = load_csv_safe(ori_path, default_dim=3)  # 可改为 4，如果四元数

    # print("读取后：", acc.shape, gyro.shape, ori.shape)

    # ====== 增强特征 ======
    acc = enhance_acc(acc)    # [T, 6]
    gyro = enhance_gyro(gyro)  # [T, 6]
    ori = enhance_ori(ori)    # [T, ori_dim*2]

    # ====== 拼接 ======
    # imu = torch.cat([acc, gyro, ori], dim=1)  # [max_frames, total_dim]
    imu = torch.cat([acc], dim=1)  # [max_frames, total_dim]
    return imu


# ------------------ 数据增强（训练集专用） ------------------
def augment_frames_and_imu(frames, imu):
    frames = frames.copy()
    imu_np = imu.numpy() if isinstance(imu, torch.Tensor) else imu.copy()
    flip_flag = False

    # 水平翻转
    if np.random.rand() < 0.5:
        flip_flag = True
        frames = frames[:, :, ::-1, :].copy()
        imu_np[...,0] = -imu_np[...,0]

    # 随机亮度
    factor = 0.9 + 0.2*np.random.rand()
    frames = (frames.astype(np.float32)*factor).clip(0,255).astype(np.uint8)

    # 高斯噪声
    noise = (np.random.randn(*frames.shape)*2.0).astype(np.float32)
    frames = (frames.astype(np.float32)+noise).clip(0,255).astype(np.uint8)

    # 随机裁剪 + 缩放
    if np.random.rand() < 0.5:
        T, H, W, C = frames.shape
        scale = np.random.uniform(0.85, 1.0)
        new_H, new_W = int(H * scale), int(W * scale)
        if new_H < H and new_W < W:
            y1 = np.random.randint(0, H - new_H + 1)
            x1 = np.random.randint(0, W - new_W + 1)
            cropped = frames[:, y1:y1 + new_H, x1:x1 + new_W, :]
            resized = np.empty_like(frames)
            for i in range(T):
                resized[i] = cv2.resize(cropped[i], (W, H), interpolation=cv2.INTER_LINEAR)
            frames = resized

    # 轻微时间扰动
    # if np.random.rand() < 0.3:
    #     idxs = np.arange(frames.shape[0])
    #     shift = np.random.randint(-2, 3)
    #     idxs = np.clip(idxs + shift, 0, frames.shape[0]-1)
    #     frames = frames[idxs]
    #     imu_np = imu_np[idxs]

    # IMU 噪声
    imu_np = imu_np + np.random.randn(*imu_np.shape).astype(np.float32)*0.01
    imu_np = imu_np * (1 + np.random.randn(*imu_np.shape)*0.01)

    return frames, torch.tensor(imu_np,dtype=torch.float32), flip_flag

# ------------------ 主预处理 ------------------
def preprocess_dataset(video_root, sensor_root, max_frames=16, cam="cam1", save_root="MMAct_preprocessed", augment_train=False):
    os.makedirs(save_root, exist_ok=True)
    action2label = {
        "carrying":0,"checking_time":1,"closing":2,"crouching":3,"entering":4,"exiting":5,
        "fall":6,"jumping":7,"kicking":8,"loitering":9,"looking_around":10,"opening":11,
        "picking_up":12,"pointing":13,"pulling":14,"pushing":15,"running":16,"setting_down":17,
        "standing":18,"talking":19,"talking_on_phone":20,"throwing":21,"transferring_object":22,
        "using_phone":23,"walking":24,"waving_hand":25
    }

    video_paths = sorted(glob.glob(os.path.join(video_root,f"subject*\\{cam}\\scene*\\session*\\*.mp4")))
    imu_modalities = ["acc_phone_clip","gyro_clip","orientation_clip"]

    def make_key(path):
        parts = path.replace("\\","/").split("/")
        subject = [p for p in parts if "subject" in p][0]
        scene = [p for p in parts if "scene" in p][0]
        session = [p for p in parts if "session" in p][0]
        name = os.path.splitext(os.path.basename(path))[0]
        return f"{subject}_{scene}_{session}_{name}"

    video_dict = {make_key(v):v for v in video_paths}
    common_keys = sorted(video_dict.keys())
    print(f"📦 找到 {len(common_keys)} 个视频样本")

    train_keys, temp_keys = train_test_split(common_keys,test_size=0.4,random_state=42)
    val_keys, test_keys = train_test_split(temp_keys,test_size=0.5,random_state=42)
    splits = {"train":train_keys,"val":val_keys,"test":test_keys}

    for split_name,key_list in splits.items():
        data_list = []
        print(f"\n🚀 正在处理 {split_name} 集 ...")
        for key in tqdm(key_list,desc=f"{split_name} processing"):
            video_path = video_dict[key]
            frames,fps,video_len = load_video_frames(video_path, resize=(112,112), max_frames=max_frames)

            imu_paths = []
            for mod in imu_modalities:
                pattern = os.path.join(sensor_root, mod, "subject*", "scene*", "session*", "*.csv")
                files = glob.glob(pattern, recursive=True)
                # print(mod, len(files), files[:3])
                found = None
                action_name = os.path.splitext(os.path.basename(video_path))[0]
                for f in files:
                    if action_name in os.path.basename(f):  # 只匹配动作名
                        found = f
                        break
                imu_paths.append(found)

            imu = align_imu_modalities(*imu_paths, video_len=video_len, fps=fps, max_frames=max_frames)

            # 数据增强仅训练集
            if split_name=="train" and augment_train:
                frames, imu, _ = augment_frames_and_imu(frames, imu)

            rgb_tensor = frames_to_normalized_tensors(frames)

            action_name = os.path.splitext(os.path.basename(video_path))[0]
            label = torch.tensor(action2label.get(action_name,-1),dtype=torch.long)

            data_list.append({
                "rgb": rgb_tensor,
                "imu": imu,
                "label": label
            })

        save_dir = os.path.join(save_root, split_name)
        save_separate_pt(data_list, save_dir)
        print(f"✅ {split_name} 集处理完成，样本数: {len(data_list)}，保存到 {save_dir}")

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    video_root = r"D:\HAR\MMAct\trimmed"
    sensor_root = r"D:\HAR\MMAct\trimmed_acc"
    preprocess_dataset(
        video_root, sensor_root,
        max_frames=16, cam="cam1",
        save_root="MMAct_preprocessed_augment_revise",
        augment_train=True
    )

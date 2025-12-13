import os
import glob
import torch
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

mp_pose = mp.solutions.pose

# ------------------ 保存 PT 文件 ------------------
def save_separate_pt(data_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, sample in enumerate(data_list):
        base_name = f"sample_{idx}"
        torch.save(sample['rgb'], os.path.join(save_dir, base_name + "_rgb.pt"))
        torch.save(sample['depth'], os.path.join(save_dir, base_name + "_depth.pt"))
        torch.save(sample['skeleton'], os.path.join(save_dir, base_name + "_skel.pt"))
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

# ------------------ 帧 -> depth ------------------
def frames_to_depth_frames(frames):
    depth_frames = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)[..., None] for f in frames], axis=0)
    return depth_frames

# ------------------ 归一化为 tensor ------------------
def frames_to_normalized_tensors(frames):
    video = frames.astype(np.float32)/255.0
    video_tensor = torch.from_numpy(video.transpose(3,0,1,2)).float()
    mean = torch.tensor([0.43216,0.394666,0.37645]).view(3,1,1,1)
    std = torch.tensor([0.22803,0.22145,0.216989]).view(3,1,1,1)
    video_tensor = (video_tensor - mean)/std
    return video_tensor

def depth_frames_to_tensor(depth_frames):
    depth = depth_frames.astype(np.float32)/255.0
    depth_tensor = torch.from_numpy(depth.transpose(3,0,1,2)).float()
    return depth_tensor

# ------------------ IMU 对齐 ------------------
def align_sensor_to_video(sensor_df, video_len, fps, max_frames=16):
    time_strs = sensor_df.iloc[:,0].astype(str).values
    imu_data = sensor_df.iloc[:,1:].values.astype(np.float32)
    t_imu = []
    for i, t in enumerate(time_strs):
        try:
            ts = pd.to_datetime(t, format="%Y%m%d_%H:%M:%S.%f")
        except:
            try:
                ts = pd.to_datetime(t, format="%Y%m%d_%H:%M:%S")
            except:
                ts = pd.Timestamp(0)+pd.to_timedelta(i/(fps if fps>0 else 30), unit='s')
        t_imu.append(ts.timestamp())
    t_imu = np.array(t_imu)
    t_video = np.arange(video_len)/(fps if fps>0 else 30)
    aligned = np.zeros((video_len, imu_data.shape[1]), dtype=np.float32)
    for j in range(imu_data.shape[1]):
        aligned[:,j] = np.interp(t_video, t_imu, imu_data[:,j])
    if video_len < max_frames:
        pad = np.tile(aligned[-1:], (max_frames - video_len,1))
        aligned = np.vstack([aligned,pad])
    else:
        aligned = aligned[:max_frames]
    return torch.tensor(aligned,dtype=torch.float32)

def align_imu_modalities(acc_path, gyro_path, ori_path, video_len, fps, max_frames=16):
    imu_list = []
    for path in [acc_path, gyro_path, ori_path]:
        if path is None or not os.path.exists(path) or os.path.getsize(path)==0:
            imu_list.append(torch.zeros((video_len,3),dtype=torch.float32))
            continue
        df = pd.read_csv(path)
        imu_list.append(align_sensor_to_video(df, video_len, fps, max_frames=video_len))
    imu = torch.cat(imu_list, dim=1)
    if imu.size(0) < max_frames:
        pad = imu[-1:].repeat(max_frames-imu.size(0),1)
        imu = torch.cat([imu,pad],dim=0)
    else:
        imu = imu[:max_frames]
    return imu

# ------------------ Skeleton 提取 ------------------
def extract_skeleton_from_frames(frames, max_frames=16, num_joints=33):
    T,H,W,C = frames.shape
    skeleton = []
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
        for t in range(T):
            frame = frames[t]
            results = pose.process(frame)
            if results.pose_landmarks:
                joints = np.array([[lm.x,lm.y,lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            else:
                joints = np.zeros((num_joints,3), dtype=np.float32)
            skeleton.append(joints)
    skeleton = np.stack(skeleton,axis=0)
    if T < max_frames:
        pad = np.tile(skeleton[-1:],(max_frames-T,1,1))
        skeleton = np.vstack([skeleton,pad])
    else:
        skeleton = skeleton[:max_frames]
    return torch.tensor(skeleton,dtype=torch.float32)

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
    # noise = (np.random.randn(*frames.shape)*5.0).astype(np.float32)
    frames = (frames.astype(np.float32)+noise).clip(0,255).astype(np.uint8)

    # ---- ✅ 改进随机裁剪 + 缩放 ----
    if np.random.rand() < 0.5:
        T, H, W, C = frames.shape
        scale = np.random.uniform(0.85, 1.0)
        new_H, new_W = int(H * scale), int(W * scale)
        if new_H < H and new_W < W:
            y1 = np.random.randint(0, H - new_H + 1)
            x1 = np.random.randint(0, W - new_W + 1)
            cropped = frames[:, y1:y1 + new_H, x1:x1 + new_W, :]
            # 使用向量化 resize，提高速度
            resized = np.empty_like(frames)
            for i in range(T):
                resized[i] = cv2.resize(cropped[i], (W, H), interpolation=cv2.INTER_LINEAR)
            frames = resized
        # 否则不裁剪（避免异常尺寸）

    # ---- 5. 轻微时间扰动 ----
    if np.random.rand() < 0.3:
        idxs = np.arange(frames.shape[0])
        shift = np.random.randint(-2, 3)
        idxs = np.clip(idxs + shift, 0, frames.shape[0]-1)
        frames = frames[idxs]

    # ---- 6. IMU 噪声 ----
    imu_np = imu_np + np.random.randn(*imu_np.shape).astype(np.float32)*0.01
    imu_np = imu_np * (1 + np.random.randn(*imu_np.shape)*0.01)

    return frames, torch.tensor(imu_np,dtype=torch.float32), flip_flag

# ------------------ swap 左右关节 ------------------
def swap_skeleton_left_right(skel_tensor):
    left_right_idx = [(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]
    skel = skel_tensor.clone()
    for l,r in left_right_idx:
        tmp = skel[:,l:l+1,:].clone()
        skel[:,l:l+1,:] = skel[:,r:r+1,:]
        skel[:,r:r+1,:] = tmp
    return skel

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
                pattern = os.path.join(sensor_root,mod,"subject*","scene*","session*","*.csv")
                files = glob.glob(pattern, recursive=True)
                found = None
                for f in files:
                    if key in os.path.basename(f):
                        found = f
                        break
                imu_paths.append(found)
            imu = align_imu_modalities(*imu_paths, video_len=video_len, fps=fps, max_frames=max_frames)

            # 数据增强仅训练集
            if split_name=="train" and augment_train:
                frames, imu, flip_flag = augment_frames_and_imu(frames, imu)
                skeleton = extract_skeleton_from_frames(frames, max_frames=max_frames, num_joints=33)
                if flip_flag:
                    skeleton = swap_skeleton_left_right(skeleton)
            else:
                skeleton = extract_skeleton_from_frames(frames, max_frames=max_frames, num_joints=33)

            depth_frames = frames_to_depth_frames(frames)
            rgb_tensor = frames_to_normalized_tensors(frames)
            depth_tensor = depth_frames_to_tensor(depth_frames)

            action_name = os.path.splitext(os.path.basename(video_path))[0]
            label = torch.tensor(action2label.get(action_name,-1),dtype=torch.long)

            data_list.append({
                "rgb": rgb_tensor,
                "depth": depth_tensor,
                "skeleton": skeleton,
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
        save_root="MMAct_preprocessed_augment",
        augment_train=True
    )

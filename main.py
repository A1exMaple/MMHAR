import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
from torch.cuda.amp import GradScaler, autocast
import gc
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# global variable
CURRENT_EPOCH = 0

# ================== 1.Dataset (unchanged) ==================
class PTFeatureDataset(Dataset):
    def __init__(self, feature_root, max_frames=16):
        self.feature_root = feature_root
        self.max_frames = max_frames
        self.rgb_files = sorted([f for f in os.listdir(feature_root) if f.endswith("_rgb.pt")])
        print(f"✅ 找到 {len(self.rgb_files)} 个样本")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.feature_root, self.rgb_files[idx])
        base_name = rgb_path.replace("_rgb.pt", "")

        imu_path = base_name + "_imu.pt"
        label_path = base_name + "_label.pt"

        rgb = torch.load(rgb_path)
        imu = torch.load(imu_path)

        # ---- label ----
        if os.path.exists(label_path):
            label = torch.load(label_path)
            label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        else:
            # 从文件名解析
            name = os.path.basename(base_name)
            label_name = name.split("_")[-1]
            label = self.label_from_name(label_name)

        return {
            "rgb": rgb.float(),
            "imu": imu.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "video_path": rgb_path
        }

    def label_from_name(self, name):
        label_dict = {
            "carrying": 0, "checking_time": 1, "closing": 2, "crouching": 3, "entering": 4, "exiting": 5,
            "fall": 6, "jumping": 7, "kicking": 8, "loitering": 9, "looking_around": 10, "opening": 11,
            "picking_up": 12, "pointing": 13, "pulling": 14, "pushing": 15, "running": 16, "setting_down": 17,
            "standing": 18, "talking": 19, "talking_on_phone": 20, "throwing": 21,
            "transferring_object": 22, "using_phone": 23, "walking": 24, "waving_hand": 25,
            "drinking": 26, "pocket_in": 27, "pocket_out": 28, "sitting": 29, "sitting_down": 30,
            "standing_up": 31, "using_pc": 32, "carrying_heavy": 33, "carrying_light": 34
        }
        return label_dict.get(name, -1)

def collate_fn_keep_imu(batch, max_frames=16):
    rgb_list, imu_list, label_list = [], [], []

    for b in batch:
        # ---------- RGB pad/truncate ----------
        v = b['rgb']  # [C, T, H, W]
        T_v = v.shape[1]
        if T_v < max_frames:
            pad_v = v[:, -1:, :, :].repeat(1, max_frames - T_v, 1, 1)
            v = torch.cat([v, pad_v], dim=1)
        else:
            v = v[:, :max_frames]
        rgb_list.append(v)

        # ---------- IMU keep original ----------
        imu_list.append(b['imu'])  # [T_i, D]

        label_list.append(b['label'])

    batch_dict = {
        "rgb": torch.stack(rgb_list, dim=0),       # [B, C, T, H, W]
        "imu": imu_list,                            # list of [T_i, D]
        "label": torch.tensor(label_list, dtype=torch.long),
        "video_path": [b['video_path'] for b in batch]
    }

    return batch_dict

def collate_fn_16(batch):
    return collate_fn_keep_imu(batch, max_frames=16)


# ================== 2. Encoders ==================
class RGBEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = video_models.r3d_18(weights="R3D_18_Weights.KINETICS400_V1")
        self.model.fc = nn.Identity()
        self.dropout = nn.Dropout(0.2)
        # ===== 冻结前 4 层（或全部 backbone），先不动预训练特征 =====
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.model(x)
        feat = self.dropout(feat)
        return feat  # [B,512]

class IMUEncoder(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=128, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_list):
        """
        x_list: list of [T_i, C], length = B
        """
        device = x_list[0].device
        lengths = torch.tensor([x.shape[0] for x in x_list], device=device)
        x_padded = pad_sequence(x_list, batch_first=True)  # [B, T_max, C]
        x = x_padded.transpose(1, 2)  # [B, C, T_max]
        x = self.relu(self.conv1(x))
        x = x.transpose(1, 2)  # [B, T_max, hidden_dim]

        # pack for LSTM
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)
        # out_unpacked: [B, T_max, hidden_dim]

        # === 时间维平均池化（mask 有效长度）===
        mask = torch.arange(out_unpacked.size(1), device=device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()

        out = (out_unpacked * mask).sum(dim=1) / lengths.unsqueeze(1)
        out = self.dropout(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: [B, C, T]
        s = x.mean(dim=2)              # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)) # [B, C]
        return x * s.unsqueeze(-1)


class GroupSEIMUEncoder(nn.Module):
    def __init__(
        self,
        in_dim=18,
        hidden_dim=96,
        lstm_layers=2,      # ← 改为 2 层
        dropout=0.2,
        groups=3            # ← IMU 常见：3 轴一组
    ):
        super().__init__()

        # ===== 3 × GroupConv =====
        self.conv1 = nn.Conv1d(
            in_dim, hidden_dim, kernel_size=3, padding=1, groups=groups
        )
        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=groups
        )
        self.conv3 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=groups
        )

        # ===== SE =====
        self.se = SEBlock(hidden_dim)

        # ===== 1 × GroupConv（SE 之后）=====
        self.conv4 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=groups
        )

        self.relu = nn.ReLU(inplace=True)

        # ===== 2 × LSTM =====
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_list):
        """
        x_list: list of [T_i, C], length = B
        """
        device = x_list[0].device
        lengths = torch.tensor([x.shape[0] for x in x_list], device=device)

        # ===== padding =====
        x_padded = pad_sequence(x_list, batch_first=True)   # [B, T, C]
        x = x_padded.transpose(1, 2)                        # [B, C, T]

        # ===== conv stack =====
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # ===== SE =====
        # x = self.se(x)

        # ===== post-SE group conv =====
        # x = self.relu(self.conv4(x))

        # ===== LSTM =====
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(
            packed_out, batch_first=True
        )

        # ===== masked temporal mean pooling =====
        mask = (
            torch.arange(out_unpacked.size(1), device=device)[None, :]
            < lengths[:, None]
        ).unsqueeze(-1).float()

        out = (out_unpacked * mask).sum(dim=1) / lengths.unsqueeze(1)
        out = self.dropout(out)

        return out

class SoftChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, C, T]
        """
        # 通道交互强度（不是重要性）
        g = self.fc(x.mean(dim=2))     # [B, C]
        return x * g.unsqueeze(-1)

class SoftChannelIMUEncoder(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=128, lstm_layers=1, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # ⭐ 新增：Soft Grouping
        self.soft_gate = SoftChannelGate(hidden_dim, reduction=4)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_list):
        """
        x_list: list of [T_i, C], length = B
        """
        device = x_list[0].device
        lengths = torch.tensor([x.shape[0] for x in x_list], device=device)

        # ===== padding =====
        x_padded = pad_sequence(x_list, batch_first=True)  # [B, T, C]
        x = x_padded.transpose(1, 2)                       # [B, C, T]

        # ===== Conv =====
        x = self.relu(self.conv1(x))

        # ⭐ Soft grouping（关键创新点）
        x = self.soft_gate(x)

        # ===== LSTM =====
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(
            packed_out, batch_first=True
        )

        # ===== masked temporal mean pooling =====
        mask = (
            torch.arange(out_unpacked.size(1), device=device)[None, :]
            < lengths[:, None]
        ).unsqueeze(-1).float()

        out = (out_unpacked * mask).sum(dim=1) / lengths.unsqueeze(1)
        out = self.dropout(out)
        return out

class AxisAwareTemporalGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        assert channels % 3 == 0, "hidden_dim must be divisible by 3 (acc/gyro/ori)"

        self.channels = channels
        self.per_axis_channels = channels // 3

        # 每个 IMU 模态一个 gate（acc / gyro / ori）
        self.fc_acc = nn.Sequential(
            nn.Linear(self.per_axis_channels * 2, self.per_axis_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.per_axis_channels // reduction, self.per_axis_channels),
            nn.Sigmoid()
        )
        self.fc_gyro = nn.Sequential(
            nn.Linear(self.per_axis_channels * 2, self.per_axis_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.per_axis_channels // reduction, self.per_axis_channels),
            nn.Sigmoid()
        )
        self.fc_ori = nn.Sequential(
            nn.Linear(self.per_axis_channels * 2, self.per_axis_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.per_axis_channels // reduction, self.per_axis_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, C, T]
        """
        B, C, T = x.shape

        # ===== temporal statistics =====
        mean = x.mean(dim=2)           # [B, C]
        std  = x.std(dim=2)            # [B, C]
        stat = torch.cat([mean, std], dim=1)  # [B, 2C]

        # ===== split by modality =====
        acc_stat  = stat[:, :2*self.per_axis_channels]
        gyro_stat = stat[:, 2*self.per_axis_channels:4*self.per_axis_channels]
        ori_stat  = stat[:, 4*self.per_axis_channels:]

        # ===== axis-aware gates =====
        g_acc  = self.fc_acc(acc_stat)
        g_gyro = self.fc_gyro(gyro_stat)
        g_ori  = self.fc_ori(ori_stat)

        # ===== concat gates =====
        g = torch.cat([g_acc, g_gyro, g_ori], dim=1)  # [B, C]

        return x * g.unsqueeze(-1)

class AxisAwareTemporalIMUEncoder(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=96, lstm_layers=1, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # ⭐ 新增：
        self.soft_gate = AxisAwareTemporalGate(hidden_dim, reduction=4)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_list):
        """
        x_list: list of [T_i, C], length = B
        """
        device = x_list[0].device
        lengths = torch.tensor([x.shape[0] for x in x_list], device=device)

        # ===== padding =====
        x_padded = pad_sequence(x_list, batch_first=True)  # [B, T, C]
        x = x_padded.transpose(1, 2)                       # [B, C, T]

        # ===== Conv =====
        x = self.relu(self.conv1(x))

        # ⭐ Soft grouping（关键创新点）
        x = self.soft_gate(x)

        # ===== LSTM =====
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(
            packed_out, batch_first=True
        )

        # ===== masked temporal mean pooling =====
        mask = (
            torch.arange(out_unpacked.size(1), device=device)[None, :]
            < lengths[:, None]
        ).unsqueeze(-1).float()

        out = (out_unpacked * mask).sum(dim=1) / lengths.unsqueeze(1)
        out = self.dropout(out)
        return out


# ================== 3. Fusion (modified) ==================
class GMUFusion(nn.Module):
    def __init__(self, feature_dim=128, dropout=0.2):
        super().__init__()
        # g = sigmoid(W * [rgb, imu])
        self.gate = nn.Linear(feature_dim * 2, feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),  # 注意这里改成 3*feature_dim
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, return_gate=False):
        rgb, imu = features
        h = torch.cat([rgb, imu], dim=-1)
        g = torch.sigmoid(self.gate(h))  # [B, D]
        fused = g * rgb + (1 - g) * imu
        enhanced = self.ffn(torch.cat([fused, h], dim=-1))
        out = self.dropout(self.norm(enhanced))
        if return_gate:
            return out, g
        else:
            return out


# ================== 4. Projectors (modified to include projection to common dim) ==================
class ContrastiveProjector(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        z = self.fc(x)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return z

# ================== 5.Model ==================
class MultiModalHAR(nn.Module):
    def __init__(self, num_classes, common_dim=128, contrastive_dim=64, IMUtype="Basic"):
        super().__init__()

        # only keep rgb + imu encoders
        self.rgb_enc = RGBEncoder()
        if IMUtype == "Basic":
            self.imu_enc = IMUEncoder()
        elif IMUtype == "GroupSE":
            self.imu_enc = GroupSEIMUEncoder()
        elif IMUtype == "SoftChannel":
            self.imu_enc = SoftChannelIMUEncoder()
        elif IMUtype == "AxisAwareTemporal":
            self.imu_enc = AxisAwareTemporalIMUEncoder()

        # project to common dim
        if IMUtype == "Basic":
            self.to_common = nn.ModuleList([
                nn.Linear(512, common_dim),
                nn.Linear(128, common_dim)
            ])
        elif IMUtype == "GroupSE":
            self.to_common = nn.ModuleList([
                nn.Linear(512, common_dim),
                nn.Linear(96, common_dim)
            ])
        elif IMUtype == "SoftChannel":
            self.to_common = nn.ModuleList([
                nn.Linear(512, common_dim),
                nn.Linear(128, common_dim)
            ])
        elif IMUtype == "AxisAwareTemporal":
            self.to_common = nn.ModuleList([
                nn.Linear(512, common_dim),
                nn.Linear(96, common_dim)
            ])

        for m in self.to_common:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        self.fusion = GMUFusion(feature_dim=common_dim, dropout=0.3)

        self.fc = nn.Sequential(
            nn.Linear(128, 256),  # 512 / 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # contrastive projectors
        self.proj_rgb = ContrastiveProjector(common_dim, contrastive_dim)
        self.proj_imu = ContrastiveProjector(common_dim, contrastive_dim)

    def forward(self, rgb, imu, mode="classification"):

        # only extract rgb+imu
        f_rgb = self.rgb_enc(rgb)      # [B,512]
        f_imu = self.imu_enc(imu)

        # project to common dim
        c_rgb = self.to_common[0](f_rgb)
        c_imu = self.to_common[1](f_imu)
        c_list = [c_rgb, c_imu]

        if mode == "classification":
            fused, gate = self.fusion(c_list, return_gate=True)
            # out = self.fc(fused)
            out = self.fc(f_imu)
            return out, gate

        elif mode == "contrastive":
            p_rgb = self.proj_rgb(c_rgb)
            p_imu = self.proj_imu(c_imu)
            return (p_rgb, p_imu)

# ================== 6.pretrain  ==================
def cross_modal_contrastive(z1, z2, labels, temperature=0.07):
    logits = torch.matmul(z1, z2.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(z1.device)
    positives = torch.sum(torch.exp(logits) * mask, dim=1)
    denominator = torch.sum(torch.exp(logits), dim=1)
    loss = -torch.log((positives + 1e-8) / (denominator + 1e-8))
    if (mask.sum(1) == 0).all():
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    return loss.mean()

def supervised_contrastive_loss(features, labels, temperature=0.15):
    device = features.device
    B = features.size(0)
    sim_matrix = torch.matmul(features, features.T) / temperature
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
    mask = mask * logits_mask
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return loss.mean()

def pairwise_mean_alignment(projs):
    means = [p.mean(dim=0) for p in projs]
    loss = 0.0
    count = 0
    for i in range(len(means)):
        for j in range(i+1, len(means)):
            loss = loss + F.mse_loss(means[i], means[j])
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=projs[0].device)
    return loss / count

def prototype_contrastive_loss(embeddings, labels, queue=None, temperature=0.07, hard_k=8):
    device = embeddings.device
    B, D = embeddings.shape
    labels_cpu = labels.cpu().numpy()
    unique_cls = sorted(list(set(labels_cpu)))
    if len(unique_cls) <= 1:
        return torch.tensor(0.0, device=device)
    # prototypes from batch
    prototypes = []
    proto_labels = []
    for c in unique_cls:
        mask = (labels == c)
        if mask.sum() == 0: continue
        proto = embeddings[mask].mean(dim=0)
        proto = proto / (proto.norm() + 1e-8)
        prototypes.append(proto)
        proto_labels.append(c)
    prototypes = torch.stack(prototypes, dim=0)  # [C, D]

    # compute similarity of each sample to each prototype
    logits_proto = torch.matmul(embeddings, prototypes.t()) / temperature  # [B, C]
    # positive index per sample:
    proto_label_to_idx = {lab: i for i, lab in enumerate(proto_labels)}
    pos_idx = torch.tensor([proto_label_to_idx[int(l.item())] for l in labels], device=device)

    # optionally add hard negatives from queue: choose top-k by similarity to embeddings
    if (queue is not None) and (queue.size(0) > 0) and hard_k > 0:
        # compute similarity with queue
        sim_q = torch.matmul(embeddings, queue.t())  # [B, Q]
        topk_vals, topk_idx = torch.topk(sim_q, k=min(hard_k, queue.size(0)), dim=1)
        # build negative logits from those topk queue elements (tiled across prototypes)
        # For simplicity, compute their exp and add to denominator only
        # convert topk to similarity logits / temperature
        # gather actual vectors
        negs = queue[topk_idx]  # [B, k, D]
        negs = negs.view(-1, D)  # [B*k, D]
        # compute logits of embeddings vs negs
        logits_negs = torch.matmul(embeddings, negs.t()) / temperature  # [B, B*k]
        # We will append these logits to denominator; to avoid huge memory, approximate by using sum_exp
        denom_proto = torch.logsumexp(logits_proto, dim=1)  # baseline denom (log)
        denom_negs = torch.logsumexp(logits_negs, dim=1)
        # positive logits
        pos_logits = logits_proto[torch.arange(B, device=device), pos_idx]
        # final loss: -log( exp(pos) / (sum_proto + sum_negs) )
        loss = - (pos_logits - torch.log( torch.exp(denom_proto) + torch.exp(denom_negs) + 1e-8 ))
        return loss.mean()
    else:
        # standard proto-softmax
        logsum = torch.logsumexp(logits_proto, dim=1)
        pos_logits = logits_proto[torch.arange(B, device=device), pos_idx]
        loss = - (pos_logits - logsum)
        return loss.mean()

class MemoryQueue:
    def __init__(self, feature_dim, queue_size=2048, device='cuda'):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.device = device
        self.registered = 0
        self.ptr = 0
        self.queue = torch.zeros(queue_size, feature_dim, device=device)
        self.labels = torch.full((queue_size,), -1, dtype=torch.long, device=device)

    @torch.no_grad()
    def enqueue(self, features, labels):
        """
        features: [B, D]
        labels: [B]
        """
        batch_size = features.size(0)
        space_left = self.queue_size - self.ptr
        if batch_size <= space_left:
            self.queue[self.ptr:self.ptr+batch_size] = features
            self.labels[self.ptr:self.ptr+batch_size] = labels
            self.ptr += batch_size
        else:
            self.queue[self.ptr:] = features[:space_left]
            self.labels[self.ptr:] = labels[:space_left]
            self.queue[:batch_size-space_left] = features[space_left:]
            self.labels[:batch_size-space_left] = labels[space_left:]
            self.ptr = batch_size - space_left
        self.registered = min(self.queue_size, self.registered + batch_size)

    def get_queue(self):
        return self.queue[:self.registered], self.labels[:self.registered]

def pretrain_contrastive_amp_supcon_with_queue(
    model, dataloader, optimizer, device,
    lambda_cross=0.5,
    lambda_supcon=0.5,
    lambda_align=0.1,
    lambda_proto=0.5,
    if_fixedlambda=False,
    align_methods=None,
    accumulation_steps=2,
    queue_size=2048,
    hard_k=8,
):
    if align_methods is None:
        align_methods = ["mean"]
    model.train()
    scaler = GradScaler()
    optimizer.zero_grad()
    total_loss = 0.0
    queues = {
        "rgb": MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        "imu": MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
    }
    for step, batch in enumerate(dataloader):
        rgb = batch['rgb'].to(device)
        # imu = batch['imu'].to(device)
        imu = [x.to(device) for x in batch["imu"]]
        labels = batch['label'].to(device)
        with torch.amp.autocast("cuda"):
            (p_rgb, p_imu) = model(rgb, imu, mode="contrastive")
            supcon_losses = []
            for name, p_feat in zip(["rgb", "imu"], [p_rgb, p_imu]):
                q_feat, q_label = queues[name].get_queue()
                if q_feat.size(0) > 0:
                    all_feat = torch.cat([p_feat, q_feat], dim=0)
                    all_label = torch.cat([labels, q_label], dim=0)
                else:
                    all_feat = p_feat
                    all_label = labels
                supcon_losses.append(supervised_contrastive_loss(all_feat, all_label))
                queues[name].enqueue(p_feat.detach(), labels)
            supcon_loss = torch.stack(supcon_losses).mean()
            loss_cross = cross_modal_contrastive(p_rgb, p_imu, labels)
            loss_align = torch.tensor(0.0, device=device)
            projs = [p_rgb, p_imu]

            if "mean" in align_methods:
                loss_align += pairwise_mean_alignment(projs)
            global_q = None
            if queues["rgb"].get_queue()[0].size(0) > 0:
                q_rgb, _ = queues["rgb"].get_queue()
                q_imu, _ = queues["imu"].get_queue()
                global_q = torch.cat([q_rgb, q_imu], dim=0)
            p_avg = (p_rgb + p_imu) / 2.0
            proto_loss = prototype_contrastive_loss(
                p_avg, labels, queue=global_q, temperature=0.07, hard_k=hard_k
            )
            losses = [loss_cross, supcon_loss, loss_align, proto_loss]
            init_lambdas = [lambda_cross, lambda_supcon, lambda_align, lambda_proto]
            # Dynamic λ update
            if 'prev_losses' not in globals():
                prev_losses = [l.item() for l in losses]
                delta_losses = [1.0 / lam if lam > 0 else 1.0 for lam in init_lambdas]
            alpha = 0.9
            eps = 1e-6
            lambda_list = []
            for i, l in enumerate(losses):
                curr = l.item()
                delta = (prev_losses[i] - curr) / (prev_losses[i] + eps)
                delta_losses[i] = alpha * delta_losses[i] + (1 - alpha) * delta
                if init_lambdas[i] == 0:
                    lam = 0.0
                else:
                    lam = init_lambdas[i] / (delta_losses[i] + eps)
                    lam = min(max(lam, 0.01), 10.0)
                lambda_list.append(lam)
                prev_losses[i] = curr
            # normalize
            s = sum(lambda_list)
            lambda_list = [l / s * len(lambda_list) for l in lambda_list]
            # final loss
            if if_fixedlambda:
                loss = sum(l * w for l, w in zip(losses, init_lambdas))
            else:
                loss = sum(l * w for l, w in zip(losses, lambda_list))
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(dataloader)

def pretrain_contrastive_supcon(
    model, dataloader, optimizer, device,
    lambda_cross=0.5,
    lambda_supcon=0.5,
    if_fixedlambda=False,
    queue_size=2048,
    warmup_queue=True,   # 新增参数，是否队列预热
):
    model.train()
    scaler = GradScaler()
    optimizer.zero_grad()
    total_loss = 0.0
    queues = {
        "rgb": MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        "imu": MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
    }
    prev_losses = None
    delta_losses = None
    alpha = 0.9
    eps = 1e-6

    for step, batch in enumerate(dataloader):
        rgb = batch['rgb'].to(device)
        imu = [x.to(device) for x in batch["imu"]]
        labels = batch['label'].to(device)

        with torch.amp.autocast("cuda"):
            (p_rgb, p_imu) = model(rgb, imu, mode="contrastive")
            # --------------------- supcon loss ---------------------
            supcon_losses = []
            queues_ready = True
            for name, p_feat in zip(["rgb", "imu"], [p_rgb, p_imu]):
                q_feat, q_label = queues[name].get_queue()

                # 如果队列没满，并且启用预热模式，只enqueue，不计算loss
                min_negatives = 128
                if warmup_queue and q_feat.size(0) < min_negatives:
                    queues[name].enqueue(p_feat.detach(), labels)
                    queues_ready = False
                    continue

                all_feat = torch.cat([p_feat, q_feat], dim=0)
                all_label = torch.cat([labels, q_label], dim=0)
                supcon_losses.append(supervised_contrastive_loss(all_feat, all_label))
                queues[name].enqueue(p_feat.detach(), labels)

            supcon_loss = torch.stack(supcon_losses).mean() if queues_ready else torch.tensor(0.0, device=device)

            # --------------------- cross-modal loss ---------------------
            loss_cross = cross_modal_contrastive(p_rgb, p_imu, labels)
            # --------------------- combine losses ---------------------
            losses = [loss_cross, supcon_loss]
            print(losses)
            init_lambdas = [lambda_cross, lambda_supcon]
            if prev_losses is None:
                prev_losses = [l.item() for l in losses]
                delta_losses = [1.0 / lam if lam > 0 else 1.0 for lam in init_lambdas]
            lambda_list = []
            for i, l in enumerate(losses):
                curr = l.item()
                delta = (prev_losses[i] - curr) / (prev_losses[i] + eps)
                delta_losses[i] = alpha * delta_losses[i] + (1 - alpha) * delta
                if init_lambdas[i] == 0:
                    lam = 0.0
                else:
                    lam = init_lambdas[i] / (delta_losses[i] + eps)
                    lam = min(max(lam, 0.01), 10.0)
                lambda_list.append(lam)
                prev_losses[i] = curr
            # normalize
            s = sum(lambda_list)
            lambda_list = [l / s * len(lambda_list) for l in lambda_list]
            # final loss
            if if_fixedlambda:
                loss = sum(l * w for l, w in zip(losses, init_lambdas))
            else:
                loss = sum(l * w for l, w in zip(losses, lambda_list))

        # ---- backward ----
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ================== 7.finetune & validation ==================
def finetune_classification(
    model, dataloader, optimizer, criterion, device, cls_only=False
):
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad()

    total_loss = 0.0
    total_acc = 0
    total_count = 0

    for step, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device)
        # imu = batch["imu"].to(device)
        imu = [x.to(device) for x in batch["imu"]]
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda"):
            # classification forward
            logits, gate = model(rgb, imu, mode="classification")

            mask = labels >= 0
            if mask.sum() == 0: continue
            logit_f = logits[mask]
            label_f = labels[mask]

            # cls loss 强制 FP32
            cls_loss = criterion(logit_f.float(), label_f)

            if cls_only:
                loss = cls_loss
            else:
                # Class-aware Gate Regularization
                gate_mean = gate.mean(dim=1)  # [B]
                target_gate = torch.full_like(gate_mean, 0.5)
                gate_loss = F.mse_loss(gate_mean, target_gate)
                lambda_gate = 0.05
                loss = cls_loss + lambda_gate * gate_loss

        scaler.scale(loss).backward()

        # optional: 防止 FP16 梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * label_f.size(0)
        total_acc += (logit_f.argmax(1) == label_f).sum().item()
        total_count += label_f.size(0)

    return total_loss / total_count, total_acc / total_count

def validate_classification(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            # imu = batch["imu"].to(device)
            imu = [x.to(device) for x in batch["imu"]]
            labels = batch["label"].to(device)

            logits, _ = model(rgb, imu, mode="classification")

            mask = labels >= 0
            if mask.sum() == 0:
                continue

            logit_f = logits[mask]
            label_f = labels[mask]

            loss = criterion(logit_f, label_f)

            total_loss += loss.item() * label_f.size(0)
            total_acc += (logit_f.argmax(1) == label_f).sum().item()
            total_count += label_f.size(0)

    return total_loss / total_count, total_acc / total_count

# ================== 8.utils ==================
def set_encoder_trainable(model, requires_grad: bool):
    # for enc in [model.rgb_enc, model.depth_enc, model.skel_enc, model.imu_enc]:
    for enc in [model.rgb_enc, model.imu_enc]:
        for p in enc.parameters():
            p.requires_grad = requires_grad

def seed_worker(worker_id):
    global CURRENT_EPOCH
    # 每个 worker 的种子 = torch initial seed + 当前 epoch
    worker_seed = (torch.initial_seed() + CURRENT_EPOCH) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ================== Main ==================
if __name__ == "__main__":
    #GPU 性能与稳定性设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    gc.collect()

    feature_root = r"D:\HAR\MMAct_preprocessed_proposed\train"
    dataset = PTFeatureDataset(feature_root, max_frames=16)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_16,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )

    val_root = r"D:\HAR\MMAct_preprocessed_proposed\val"
    val_dataset = PTFeatureDataset(val_root, max_frames=16)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_16,
        num_workers=8,
        pin_memory=True
    )

    #模型 & 损失
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalHAR(num_classes=35, common_dim=128, contrastive_dim=64, IMUtype="SoftChannel").to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ================== 1.pretrain ==================
    # if_pretrain = True
    if_pretrain = False
    if if_pretrain:
        num_pretrain_epochs = 50
        for epoch in range(num_pretrain_epochs):
            # === 前 5 epoch IMU warmup ===
            if epoch < 5:
                for param in model.rgb_enc.parameters():
                    param.requires_grad = False
                for param in model.imu_enc.parameters():
                    param.requires_grad = True
            elif epoch == 5:
                # 解冻 RGB，同时设置不同 LR
                for param in model.rgb_enc.parameters():
                    param.requires_grad = True

            optimizer1 = torch.optim.Adam([
                {"params": model.rgb_enc.parameters(), "lr": 3e-5},
                {"params": model.imu_enc.parameters(), "lr": 3e-3},
                {"params": model.proj_rgb.parameters(), "lr": 2e-4},
                {"params": model.proj_imu.parameters(), "lr": 2e-4},
                {"params": model.to_common.parameters(), "lr": 2e-4},
            ], weight_decay=1e-4)

            # loss = pretrain_contrastive_amp_supcon_with_queue(
            #     model, dataloader, optimizer1, device,
            #     lambda_cross=0.35, lambda_supcon=0.2,
            #     lambda_align=0.05, lambda_proto=0.8, if_fixedlambda=False,
            #     align_methods=['mean'],
            #     accumulation_steps=accumulation_steps,
            #     queue_size=2048,
            #     hard_k=8
            # )

            loss = pretrain_contrastive_supcon(
                model, dataloader, optimizer1, device,
                lambda_cross=0.35, lambda_supcon=0.2,
                if_fixedlambda=True, queue_size=2048,
            )

            print(f"[Pretrain] Epoch {epoch}: loss={loss:.4f}")
            torch.save(model.state_dict(), f"pretrain/pretrain_epoch_{epoch}.pth")
    # else:
    #     pretrain_path = "pretrain/pretrain_epoch_49.pth"
    #     checkpoint = torch.load(pretrain_path, map_location=device)
    #     model_dict = model.state_dict()
    #     filtered_ckpt = {k: v for k, v in checkpoint.items()
    #                      if k in model_dict and not (k.startswith("fusion.") or k.startswith("fc."))}
    #     model_dict.update(filtered_ckpt)
    #     model.load_state_dict(model_dict, strict=False)

    # ================== 2.Finetune ==================
    latest_model_path = "multimodal_har_model_latest.pth"
    best_model_path = "multimodal_har_model_best.pth"
    best_val_acc = 0.0
    num_finetune_epochs = 100
    val_interval = 3

    for p in model.rgb_enc.parameters():
        p.requires_grad = False
    for p in model.imu_enc.parameters():
        p.requires_grad = True
    for p in model.to_common.parameters():
        p.requires_grad = False
    for p in model.fusion.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.rgb_enc.parameters(), "lr": 3e-5},
        {"params": model.imu_enc.parameters(), "lr": 3e-3},
        {"params": model.to_common.parameters(), "lr": 2e-4},
        {"params": model.fusion.parameters(), "lr": 2e-4},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    for epoch in range(num_finetune_epochs):
        loss, acc = finetune_classification(
            model, dataloader, optimizer, criterion, device, cls_only=True
        )
        print(f"[Finetune] Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")
        scheduler.step()

        # ================== 3.validate ==================
        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = validate_classification(model, val_loader, criterion, device)
            print(f"[Validation] Epoch {epoch}: loss={val_loss:.4f}, acc={val_acc:.4f}")

            torch.save(model.state_dict(), latest_model_path)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"✨ 保存最佳模型: Epoch {epoch}, val_acc={val_acc:.4f}")

    print(f"✅ 训练结束！最佳验证准确率: {best_val_acc:.4f}")
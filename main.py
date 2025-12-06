import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
from torch.cuda.amp import GradScaler, autocast
import gc
import torch.utils.checkpoint as checkpoint
import collections
import numpy as np
import random

# global variable
CURRENT_EPOCH = 0

# ================== Dataset (unchanged) ==================
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
        depth_path = base_name + "_depth.pt"
        skel_path = base_name + "_skel.pt"
        imu_path  = base_name + "_imu.pt"
        label_path = base_name + "_label.pt"

        rgb = torch.load(rgb_path)
        depth = torch.load(depth_path)
        skel = torch.load(skel_path)
        imu = torch.load(imu_path)

        if os.path.exists(label_path):
            label = torch.load(label_path)
            label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        else:
            name = os.path.basename(base_name)
            label_name = name.split("_")[-1]
            label = self.label_from_name(label_name)

        return {
            "rgb": rgb.float(),
            "depth": depth.float(),
            "skeleton": skel.float(),
            "imu": imu.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "video_path": rgb_path
        }

    def label_from_name(self, name):
        label_dict = {
            "carrying":0, "checking_time":1, "closing":2, "crouching":3,
            "entering":4, "exiting":5, "fall":6, "jumping":7, "kicking":8,
            "loitering":9, "looking_around":10, "opening":11, "picking_up":12,
            "pointing":13, "pulling":14, "pushing":15, "running":16,
            "setting_down":17, "standing":18, "talking":19, "talking_on_phone":20,
            "throwing":21, "transferring_object":22, "using_phone":23,
            "walking":24, "waving_hand":25
        }
        return label_dict.get(name, -1)

def collate_fn(batch, max_frames=16):
    rgb_list, depth_list, skel_list, imu_list, label_list = [], [], [], [], []
    for b in batch:
        v = b['rgb']
        t_v = v.shape[1]
        if t_v < max_frames:
            pad_v = v[:, -1:, :, :].repeat(1, max_frames - t_v, 1, 1)
            v = torch.cat([v, pad_v], dim=1)
        else:
            v = v[:, :max_frames, :, :]
        rgb_list.append(v)

        d = b.get('depth', torch.zeros(1, max_frames, v.shape[2], v.shape[3], dtype=v.dtype))
        t_d = d.shape[1]
        if t_d < max_frames:
            pad_d = d[:, -1:, :, :].repeat(1, max_frames - t_d, 1, 1)
            d = torch.cat([d, pad_d], dim=1)
        else:
            d = d[:, :max_frames, :, :]
        depth_list.append(d)

        skel = b.get('skeleton', torch.zeros(max_frames, 33, 3, dtype=v.dtype))
        t_s = skel.shape[0]
        if t_s < max_frames:
            pad_s = skel[-1:].repeat(max_frames - t_s, 1, 1)
            skel = torch.cat([skel, pad_s], dim=0)
        else:
            skel = skel[:max_frames]
        skel_list.append(skel)

        imu = b['imu']
        t_i = imu.shape[0]
        if t_i < max_frames:
            pad_i = imu[-1:].repeat(max_frames - t_i, 1)
            imu = torch.cat([imu, pad_i], dim=0)
        else:
            imu = imu[:max_frames]
        imu_list.append(imu)

        label_list.append(b['label'])

    batch_dict = {
        "rgb": torch.stack(rgb_list, dim=0),
        "depth": torch.stack(depth_list, dim=0),
        "skeleton": torch.stack(skel_list, dim=0),
        "imu": torch.stack(imu_list, dim=0),
        "label": torch.tensor(label_list, dtype=torch.long),
        "video_path": [b['video_path'] for b in batch]
    }
    return batch_dict

def collate_fn_16(batch):
    return collate_fn(batch, max_frames=16)

# ================== 2. Encoders（沿用） ==================
# class RGBEncoder(nn.Module):
#     def __init__(self, pretrained=False):
#         super().__init__()
#         self.model = video_models.r3d_18(weights="R3D_18_Weights.KINETICS400_V1")
#         self.model.fc = nn.Identity()
#
#         # 统一 stem
#         if isinstance(self.model.stem, collections.OrderedDict):
#             self.model.stem = nn.Sequential(*self.model.stem.values())
#         elif isinstance(self.model.stem, nn.Sequential):
#             if isinstance(self.model.stem[0], collections.OrderedDict):
#                 self.model.stem = nn.Sequential(*self.model.stem[0].values())
#
#         # 统一 layer1–4
#         for i in range(1, 5):
#             layer = getattr(self.model, f"layer{i}")
#             if isinstance(layer, collections.OrderedDict):
#                 setattr(self.model, f"layer{i}", nn.Sequential(*layer.values()))
#
#         self.dropout = nn.Dropout(0.3)
#
#         # 冻结参数
#         for name, param in self.model.named_parameters():
#             param.requires_grad = False
#
#     def forward(self, x):
#         from torch.utils.checkpoint import checkpoint
#
#         # 判断是否有可训练参数
#         trainable = any(p.requires_grad for p in self.model.parameters())
#
#         if trainable and self.training:
#             # 训练阶段且有可训练参数时使用 checkpoint
#             x = checkpoint(lambda y: self.model.stem(y), x)
#             x = checkpoint(lambda y: self.model.layer1(y), x)
#             x = checkpoint(lambda y: self.model.layer2(y), x)
#             x = checkpoint(lambda y: self.model.layer3(y), x)
#             x = checkpoint(lambda y: self.model.layer4(y), x)
#         else:
#             # 冻结阶段直接 forward
#             x = self.model.stem(x)
#             x = self.model.layer1(x)
#             x = self.model.layer2(x)
#             x = self.model.layer3(x)
#             x = self.model.layer4(x)
#
#         # 全局平均池化 + dropout
#         x = x.mean(dim=[2, 3, 4])
#         x = self.dropout(x)
#         return x


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

# Depth encoder -> out_dim 64
class DepthEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.Conv3d(32, out_dim, kernel_size=1),
            nn.AdaptiveAvgPool3d(1)
        )
    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)  # [B, out_dim]

# class DepthEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
#             nn.ReLU(),
#             nn.Dropout3d(0.2),
#             nn.AdaptiveAvgPool3d(1)
#         )
#     def forward(self, x):
#         return self.cnn(x).view(x.size(0), -1)  # [B,16]

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x, adj):
        B = x.size(0)
        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0).expand(B, -1, -1)
        else:
            adj_b = adj
        h = torch.bmm(adj_b, x)
        h = self.fc(h)
        return h

class SkeletonGCNEncoderLSTM(nn.Module):
    def __init__(self, num_joints=33, in_dim=3, hidden_dim=64, lstm_hidden=128, dropout=0.2):
        super().__init__()
        # NOTE: you can improve adj here with domain knowledge later
        self.register_buffer("adj", torch.eye(num_joints))
        self.gcn1 = GraphConv(in_dim, 32)
        self.gcn2 = GraphConv(32, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, J, D = x.shape
        feats = []
        for t in range(T):
            xt = x[:, t]
            h = F.relu(self.gcn1(xt, self.adj))
            h = self.gcn2(h, self.adj)
            h = h.mean(dim=1)
            feats.append(h)
        feats = torch.stack(feats, dim=1)
        _, (h_n, _) = self.lstm(feats)
        out = h_n.squeeze(0)
        out = self.dropout(out)
        return out  # [B, lstm_hidden]


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=1)  # [B, C]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.unsqueeze(1)

class IMUEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=66, lstm_layers=1, groups=3, dropout=0.2):
    # def __init__(self, in_dim=9, hidden_dim=33, lstm_layers=1, groups=3, dropout=0.2):
        super().__init__()
        self.groups = groups
        self.group_conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1, groups=groups)
        self.se = SEModule(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)          # [B, C, T]
        x = F.relu(self.group_conv(x)) # [B, hidden_dim, T]
        x = x.transpose(1, 2)          # [B, T, hidden_dim]
        x = self.se(x)
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]                  # [B, hidden_dim]
        out = self.dropout(out)
        return out

# ================== 3. Fusion (modified) ==================
# Simple Transformer-based fusion that expects all modality features to be same dim
class SimpleTransformerFusion(nn.Module):
    def __init__(self, feature_dim=128, nhead=4, num_layers=1, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.5)  # 加 dropout
    def forward(self, features):
        # features: list of [B, D] where D == feature_dim
        proj_stack = torch.stack(features, dim=1)  # [B, M, D]
        out = self.transformer(proj_stack)        # [B, M, D]
        fused = out.mean(dim=1)                   # [B, D]
        fused = self.norm(fused)
        fused = self.dropout(fused)  # dropout
        return fused

class EnhancedTransformerFusion(nn.Module):
    def __init__(self, feature_dim=128, nhead=8, num_layers=2, dropout=0.2, attn_pool=True):
        """
        feature_dim: 每个 modality 的特征维度
        nhead: 多头注意力头数
        num_layers: Transformer Encoder 层数
        dropout: Transformer 内部 dropout
        attn_pool: 是否使用 learnable attention pooling 替代 mean pooling
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.2)
        self.attn_pool = attn_pool

        if attn_pool:
            # Learnable attention pooling参数
            self.pool_attn = nn.Linear(feature_dim, 1)

        # 可选的 FFN 用于增强非线性
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features):
        """
        features: list of [B, D]  每个 modality 的特征
        """
        proj_stack = torch.stack(features, dim=1)  # [B, M, D]
        out = self.transformer(proj_stack)         # [B, M, D]

        # Learnable attention pooling
        if self.attn_pool:
            attn_scores = self.pool_attn(out)             # [B, M, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, M, 1]
            fused = (out * attn_weights).sum(dim=1)      # [B, D]
        else:
            fused = out.mean(dim=1)                       # [B, D]

        # 加上原始 mean 残差
        fused = fused + proj_stack.mean(dim=1)

        # FFN + LayerNorm + Dropout
        fused = self.ffn(fused)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused
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

class CompletionProjector(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        return self.fc(x)

class CompletionAttention(nn.Module):
    def __init__(self, latent_dim=64, nhead=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, target_feat, source_feats):
        B = target_feat.size(0)
        if target_feat is None:
            query = torch.zeros(B, 1, self.latent_dim, device=source_feats[0].device)
        else:
            query = self.query_proj(target_feat).unsqueeze(1)
        key_value = torch.stack(source_feats, dim=1)
        out, _ = self.attn(query=query, key=key_value, value=key_value)
        out = out.squeeze(1)
        out = self.mlp(out)
        return out

# ================== 5. Full Model (modified) ==================
class MultiModalHAR(nn.Module):
    def __init__(self, num_classes, common_dim=128, contrastive_dim=64, completion_confidence_scale=0.7):
        super().__init__()
        # encoders
        self.rgb_enc = RGBEncoder()
        self.depth_enc = DepthEncoder()
        self.skel_enc = SkeletonGCNEncoderLSTM()
        self.imu_enc = IMUEncoder()

        # ===== new: project each encoder output to COMMON dim (to avoid RGB domination) =====
        self.to_common = nn.ModuleList([
            nn.Linear(512, common_dim),   # rgb
            nn.Linear(64, common_dim),    # depth
            nn.Linear(128, common_dim),   # skel (ensure match your skel output)
            nn.Linear(66, common_dim)     # imu
        ])

        # self.to_common = nn.ModuleList([
        #     nn.Linear(512, common_dim),   # rgb
        #     nn.Linear(16, common_dim),    # depth
        #     nn.Linear(128, common_dim),   # skel (ensure match your skel output)
        #     nn.Linear(33, common_dim)     # imu
        # ])
        for m in self.to_common:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        # fusion expects common_dim features
        # self.fusion = SimpleTransformerFusion(feature_dim=common_dim, nhead=4, num_layers=1, dropout=0.2)
        self.fusion = EnhancedTransformerFusion(feature_dim=common_dim, nhead=8, num_layers=2, dropout=0.1, attn_pool=True)

        # classification head
        # self.fc = nn.Sequential(
        #     nn.Linear(common_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.6),
        #     nn.Linear(128, num_classes)
        # )

        self.fc = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # contrastive/completion projectors operate on common-dimensional features
        self.proj_rgb   = ContrastiveProjector(common_dim, contrastive_dim)
        self.proj_depth = ContrastiveProjector(common_dim, contrastive_dim)
        self.proj_skel  = ContrastiveProjector(common_dim, contrastive_dim)
        self.proj_imu   = ContrastiveProjector(common_dim, contrastive_dim)

        self.cproj_rgb   = CompletionProjector(common_dim, contrastive_dim)
        self.cproj_depth = CompletionProjector(common_dim, contrastive_dim)
        self.cproj_skel  = CompletionProjector(common_dim, contrastive_dim)
        self.cproj_imu   = CompletionProjector(common_dim, contrastive_dim)

        self.completion_attentions = nn.ModuleList([
            CompletionAttention(latent_dim=contrastive_dim, nhead=4) for _ in range(4)
        ])

        # per-modality classification head (on contrastive proj) for modality confidence
        self.modality_clf = nn.ModuleList([nn.Linear(contrastive_dim, num_classes) for _ in range(4)])

        self.num_classes = num_classes
        self.completion_confidence_scale = completion_confidence_scale

    def forward(self, rgb, depth, skel, imu, mode="classification", do_completion=False):
        # raw encoder outputs
        f_rgb = self.rgb_enc(rgb)    # [B,512]
        f_depth = self.depth_enc(depth)  # [B,16]
        f_skel = self.skel_enc(skel)     # [B,128]
        f_imu = self.imu_enc(imu)        # [B,33]

        # project to common dim
        c_rgb = self.to_common[0](f_rgb)
        c_depth = self.to_common[1](f_depth)
        c_skel = self.to_common[2](f_skel)
        c_imu = self.to_common[3](f_imu)

        if mode == "classification":
            fused = self.fusion([c_rgb, c_depth, c_skel, c_imu])  # [B,common_dim]
            out = self.fc(fused)

            # per-modality contrastive proj for modality confidence
            p_rgb = self.proj_rgb(c_rgb)
            p_depth = self.proj_depth(c_depth)
            p_skel = self.proj_skel(c_skel)
            p_imu = self.proj_imu(c_imu)
            p_list = [p_rgb, p_depth, p_skel, p_imu]

            per_modality_logits = [clf(p) for clf, p in zip(self.modality_clf, p_list)]
            per_modality_probs = [F.softmax(l, dim=1) for l in per_modality_logits]
            per_modality_conf = [torch.max(p, dim=1).values for p in per_modality_probs]

            if do_completion:
                per_modality_conf = [c * self.completion_confidence_scale for c in per_modality_conf]

            return out, per_modality_conf, per_modality_logits

        elif mode in ("contrastive", "contrastive_with_completion"):
            p_rgb   = self.proj_rgb(c_rgb)
            p_depth = self.proj_depth(c_depth)
            p_skel  = self.proj_skel(c_skel)
            p_imu   = self.proj_imu(c_imu)

            cc_rgb   = self.cproj_rgb(c_rgb)
            cc_depth = self.cproj_depth(c_depth)
            cc_skel  = self.cproj_skel(c_skel)
            cc_imu   = self.cproj_imu(c_imu)

            completion_outputs = None
            if mode == "contrastive_with_completion" or do_completion:
                c_list = [cc_rgb, cc_depth, cc_skel, cc_imu]
                completed = []
                for i in range(4):
                    sources = [c_list[j] for j in range(4) if j != i]
                    completed_i = self.completion_attentions[i](c_list[i], sources)
                    completed.append(completed_i)
                completion_outputs = completed

            return (p_rgb, p_depth, p_skel, p_imu), (cc_rgb, cc_depth, cc_skel, cc_imu), completion_outputs

# ================== Loss functions (unchanged mostly) ==================
def intra_modal_contrastive(z, labels, temperature=0.07):
    logits = torch.matmul(z, z.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(z.device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=z.device)
    mask = mask * logits_mask
    positives = torch.sum(torch.exp(logits) * mask, dim=1)
    denominator = torch.sum(torch.exp(logits) * logits_mask, dim=1)
    loss = -torch.log((positives + 1e-8) / (denominator + 1e-8))
    if (mask.sum(1) == 0).all():
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    return loss.mean()

def cross_modal_contrastive(z1, z2, labels, temperature=0.07):
    logits = torch.matmul(z1, z2.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(z1.device)
    positives = torch.sum(torch.exp(logits) * mask, dim=1)
    denominator = torch.sum(torch.exp(logits), dim=1)
    loss = -torch.log((positives + 1e-8) / (denominator + 1e-8))
    if (mask.sum(1) == 0).all():
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    return loss.mean()

def cosine_sim(a, b, eps=1e-8):
    a_n = a / (a.norm(dim=1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=1)

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

# Alignment & MMD & coral (unchanged)
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

def coral_loss(x, y):
    d = x.size(1)
    xm = x - x.mean(dim=0, keepdim=True)
    ym = y - y.mean(dim=0, keepdim=True)
    cx = (xm.t() @ xm) / (x.size(0) - 1)
    cy = (ym.t() @ ym) / (y.size(0) - 1)
    loss = F.mse_loss(cx, cy) / (4 * d * d)
    return loss

def pairwise_coral_alignment(projs):
    loss = 0.0
    count = 0
    for i in range(len(projs)):
        for j in range(i+1, len(projs)):
            loss = loss + coral_loss(projs[i], projs[j])
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=projs[0].device)
    return loss / count

def _rbf_kernel(x, y, sigma):
    x2 = (x*x).sum(dim=1, keepdim=True)
    y2 = (y*y).sum(dim=1, keepdim=True)
    dist2 = x2 - 2 * (x @ y.t()) + y2.t()
    k = torch.exp(-dist2 / (2 * (sigma**2)))
    return k

def mmd_rbf(x, y, sigmas=(1.0, 2.0, 4.0)):
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for s in sigmas:
        Kxx = _rbf_kernel(x, x, s)
        Kyy = _rbf_kernel(y, y, s)
        Kxy = _rbf_kernel(x, y, s)
        nx = x.size(0)
        ny = y.size(0)
        xx += (Kxx.sum() - torch.diag(Kxx).sum()) / (nx * (nx - 1) + 1e-8)
        yy += (Kyy.sum() - torch.diag(Kyy).sum()) / (ny * (ny - 1) + 1e-8)
        xy += Kxy.sum() / (nx * ny + 1e-8)
    return xx + yy - 2.0 * xy

def pairwise_mmd_alignment(projs, sigmas=(1.0,2.0,4.0)):
    loss = 0.0
    count = 0
    for i in range(len(projs)):
        for j in range(i+1, len(projs)):
            loss = loss + mmd_rbf(projs[i], projs[j], sigmas=sigmas)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=projs[0].device)
    return loss / count

# ================== Training utilities (unchanged with small mods) ==================
# ================== MemoryQueue (带 label) ==================
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

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, verbose=True):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return
        elif loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def set_encoder_trainable(model, requires_grad: bool):
    for enc in [model.rgb_enc, model.depth_enc, model.skel_enc, model.imu_enc]:
        for p in enc.parameters():
            p.requires_grad = requires_grad

def freeze_encoder_layers_auto(model, encoder_name, unfreeze_until_ratio=0.0):
    encoder = getattr(model, encoder_name)
    params = list(encoder.parameters())
    total = len(params)
    unfreeze_until = int(total * unfreeze_until_ratio)
    for i, p in enumerate(params):
        p.requires_grad = True if i < unfreeze_until else False

def unfreeze_all_encoders(model):
    for enc_name in ["rgb_enc", "depth_enc", "skel_enc", "imu_enc"]:
        for p in getattr(model, enc_name).parameters():
            p.requires_grad = True

def seed_worker(worker_id):
    global CURRENT_EPOCH
    # 每个 worker 的种子 = torch initial seed + 当前 epoch
    worker_seed = (torch.initial_seed() + CURRENT_EPOCH) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============== New: prototype + hard-negative helper ==============
def prototype_contrastive_loss(embeddings, labels, queue=None, temperature=0.07, hard_k=8):
    """
    embeddings: [B, D] normalized (contrastive space)
    labels: [B]
    queue: optional tensor [Q, D] normalized
    For each class in batch compute prototype (mean), then do a softmax over prototypes + selected hard negatives from queue.
    """
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

def params_with_grad(model):
    return [p for n, p in model.named_parameters()
                   if p.requires_grad and not any(h in n for h in [
                       "proj_rgb", "proj_depth", "proj_skel", "proj_imu", "to_common", "fusion"
                   ])]

# ================== Training loops (modified pretrain to include prototypes + hard negatives) ==================
def pretrain_contrastive_amp_supcon_with_queue(model, dataloader, optimizer, device,
                                               lambda_cross=0.5,
                                               lambda_supcon=0.5,
                                               lambda_completion_recon=1.0,
                                               lambda_completion_struct=0.5,
                                               lambda_align=0.1,
                                               lambda_proto=0.5,
                                               if_fixedlambda=False,
                                               align_methods=None,
                                               accumulation_steps=2,
                                               queue_size=2048,
                                               hard_k=8):
    """
    Modified: compute prototype loss + hard negatives from queue.
    """
    if align_methods is None:
        align_methods = ['mean']

    model.train()
    scaler = GradScaler()
    total_loss = 0.0
    optimizer.zero_grad()

    # queues per modality
    queues = {
        'rgb': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        'depth': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        'skel': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        'imu': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device)
    }

    # NOTE: We'll keep queue persistent across batches inside this epoch call
    for step, batch in enumerate(dataloader):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        skel = batch['skeleton'].to(device)
        imu = batch['imu'].to(device)
        labels = batch['label'].to(device)

        with torch.amp.autocast('cuda'):
            (p_rgb, p_depth, p_skel, p_imu), (c_rgb, c_depth, c_skel, c_imu), completion_outputs = \
                model(rgb, depth, skel, imu, mode="contrastive_with_completion", do_completion=True)

            # intra/cross/supcon over p_*
            intra_losses = []
            supcon_losses = []
            for name, p_feat in zip(['rgb','depth','skel','imu'], [p_rgb,p_depth,p_skel,p_imu]):
                q_feats, q_labels = queues[name].get_queue()
                if q_feats.size(0) > 0:
                    all_feats = torch.cat([p_feat, q_feats], dim=0)
                    all_labels = torch.cat([labels, q_labels], dim=0)
                else:
                    all_feats = p_feat
                    all_labels = labels

                intra_loss_i = intra_modal_contrastive(all_feats, all_labels)
                supcon_loss_i = supervised_contrastive_loss(all_feats, all_labels)  # might ignore -1 labels

                intra_losses.append(intra_loss_i)
                supcon_losses.append(supcon_loss_i)

                queues[name].enqueue(p_feat.detach(), labels)

            cross_losses_list = [
                cross_modal_contrastive(p_rgb, p_depth, labels),
                cross_modal_contrastive(p_rgb, p_skel, labels),
                cross_modal_contrastive(p_rgb, p_imu, labels),
                cross_modal_contrastive(p_depth, p_skel, labels),
                cross_modal_contrastive(p_depth, p_imu, labels),
                cross_modal_contrastive(p_skel, p_imu, labels)
            ]
            loss_intra = torch.stack(intra_losses).mean()
            loss_cross = torch.stack(cross_losses_list).mean()
            supcon_loss = torch.stack(supcon_losses).mean()

            # completion losses
            loss_completion_recon = torch.tensor(0.0, device=device)
            loss_completion_struct = torch.tensor(0.0, device=device)
            if completion_outputs is not None:
                completed = completion_outputs
                real_c = [c_rgb, c_depth, c_skel, c_imu]
                recon_losses = [1.0 - cosine_sim(comp, real).mean() for comp, real in zip(completed, real_c)]
                loss_completion_recon = torch.stack(recon_losses).mean()

                struct_losses = []
                for i in range(4):
                    comp_i = completed[i]
                    real_i = real_c[i]
                    for j in range(4):
                        if j == i: continue
                        struct_losses.append(F.mse_loss(cosine_sim(comp_i, real_c[j]),
                                                        cosine_sim(real_i, real_c[j])))
                if len(struct_losses) > 0:
                    loss_completion_struct = torch.stack(struct_losses).mean()

            # alignment loss 把不同模态的特征空间对齐，避免 cross-modal 冲突。
            loss_align = torch.tensor(0.0, device=device)
            projs = [p_rgb, p_depth, p_skel, p_imu]
            if 'mean' in align_methods:
                loss_align = loss_align + pairwise_mean_alignment(projs)
            if 'coral' in align_methods:
                loss_align = loss_align + pairwise_coral_alignment(projs)
            if 'mmd' in align_methods:
                loss_align = loss_align + pairwise_mmd_alignment(projs)

            # prototype loss (new)  增强同类样本的聚类能力，同时利用队列中的 hard negative 来增加判别力。
            # use concatenated p_all to compute prototypes per class in current batch
            # for hard negatives use concatenated queue across modalities (simple concat)
            # Build a queue for protos
            global_queue = torch.cat([queues['rgb'].get_queue()[0],
                                      queues['depth'].get_queue()[0],
                                      queues['skel'].get_queue()[0],
                                      queues['imu'].get_queue()[0]], dim=0) if (queues['rgb'].get_queue()[0].size(0)>0) else None
            # compute prototype loss on fused set of modalities by averaging proj per sample
            p_avg = (p_rgb + p_depth + p_skel + p_imu) / 4.0
            proto_loss = prototype_contrastive_loss(p_avg, labels, queue=global_queue, temperature=0.07, hard_k=hard_k)

            # losses = [loss_cross, supcon_loss, loss_completion_recon,
            #           loss_completion_struct, loss_align, proto_loss]
            # init_lambdas = [lambda_cross, lambda_supcon, lambda_completion_recon,
            #                 lambda_completion_struct, lambda_align, lambda_proto]  # 根据你的偏好设置初始值

            losses = [loss_cross, supcon_loss, loss_align, proto_loss]
            init_lambdas = [lambda_cross, lambda_supcon, lambda_align, lambda_proto]  # 根据你的偏好设置初始值

            # 打印 losses 列表里的 item
            # print([l.item() for l in losses])

            # 上一轮记录的 loss，初始化为当前值
            if 'prev_losses' not in globals():
                prev_losses = [l.item() for l in losses]
                # delta_losses 初始为 1.0，但可乘以 init_lambdas 调整比例
                delta_losses = []
                for lam in init_lambdas:
                    if lam == 0.0:
                        delta_losses.append(1.0)  # 固定loss不动态更新
                    else:
                        delta_losses.append(1.0 / lam)  # 非零loss按原逻辑

            alpha = 0.9  #平滑系数 alpha越大 lambda更新越平稳
            epsilon = 1e-6

            lambda_list = []
            for i, l in enumerate(losses):
                curr = l.item()
                delta = (prev_losses[i] - curr) / (prev_losses[i] + epsilon)  # 变化率
                delta_losses[i] = alpha * delta_losses[i] + (1 - alpha) * delta
                # 限制范围避免梯度爆炸
                if init_lambdas[i] == 0.0:
                    lambda_i = 0.0
                else:
                    lambda_i = init_lambdas[i] / (delta_losses[i] + epsilon)
                    lambda_i = max(min(lambda_i, 10.0), 0.01)
                lambda_list.append(lambda_i)
                prev_losses[i] = curr  # 更新上一轮 loss

            # 可归一化
            sum_lambda = sum(lambda_list)
            lambda_list = [l / sum_lambda * len(lambda_list) for l in lambda_list]


            if if_fixedlambda:
                # 静态lambda
                loss = sum(l * w for l, w in zip(losses, init_lambdas))
            else:
                # 动态lambda
                loss = sum(l * lam for l, lam in zip(losses, lambda_list))

            loss = loss / accumulation_steps

        # === AMP backward & unscale -> gradient clipping ===
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            # unscale before clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)

# ================== Finetune loops (minor: unscale + clip) ==================
def finetune_classification_amp(model, dataloader, optimizer, criterion, device, accumulation_steps=2,
                                align_methods=None,
                                queue_size=2048,
                                hard_k=8,
                                lambda_cls=1,
                                lambda_supcon=0.5,
                                lambda_align=0.1,
                                lambda_proto=0.5,
                                if_fixedlambda=False,
                                ):
    model.train()
    scaler = GradScaler()
    total_loss, total_acc = 0.0, 0.0
    valid_count = 0
    optimizer.zero_grad()

    """
    Modified: compute prototype loss + hard negatives from queue.
    """
    if align_methods is None:
        align_methods = ['mean']

    # queues per modality
    queues = {
        'rgb': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        'depth': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        'skel': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device),
        'imu': MemoryQueue(feature_dim=64, queue_size=queue_size, device=device)
    }

    for step, batch in enumerate(dataloader):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        skel = batch['skeleton'].to(device)
        imu = batch['imu'].to(device)
        labels = batch['label'].to(device)

        with torch.amp.autocast('cuda'):
            (p_rgb, p_depth, p_skel, p_imu), (c_rgb, c_depth, c_skel, c_imu), completion_outputs = \
                model(rgb, depth, skel, imu, mode="contrastive_with_completion", do_completion=True)

            # intra/cross/supcon over p_*
            # intra_losses = []
            supcon_losses = []
            for name, p_feat in zip(['rgb', 'depth', 'skel', 'imu'], [p_rgb, p_depth, p_skel, p_imu]):
                q_feats, q_labels = queues[name].get_queue()
                if q_feats.size(0) > 0:
                    all_feats = torch.cat([p_feat, q_feats], dim=0)
                    all_labels = torch.cat([labels, q_labels], dim=0)
                else:
                    all_feats = p_feat
                    all_labels = labels

                # intra_loss_i = intra_modal_contrastive(all_feats, all_labels)
                supcon_loss_i = supervised_contrastive_loss(all_feats, all_labels)  # might ignore -1 labels

                # intra_losses.append(intra_loss_i)
                supcon_losses.append(supcon_loss_i)

                queues[name].enqueue(p_feat.detach(), labels)

            # cross_losses_list = [
            #     cross_modal_contrastive(p_rgb, p_depth, labels),
            #     cross_modal_contrastive(p_rgb, p_skel, labels),
            #     cross_modal_contrastive(p_rgb, p_imu, labels),
            #     cross_modal_contrastive(p_depth, p_skel, labels),
            #     cross_modal_contrastive(p_depth, p_imu, labels),
            #     cross_modal_contrastive(p_skel, p_imu, labels)
            # ]
            # loss_intra = torch.stack(intra_losses).mean()
            # loss_cross = torch.stack(cross_losses_list).mean()
            supcon_loss = torch.stack(supcon_losses).mean()


            # alignment loss 把不同模态的特征空间对齐，避免 cross-modal 冲突。
            loss_align = torch.tensor(0.0, device=device)
            projs = [p_rgb, p_depth, p_skel, p_imu]
            if 'mean' in align_methods:
                loss_align = loss_align + pairwise_mean_alignment(projs)
            if 'coral' in align_methods:
                loss_align = loss_align + pairwise_coral_alignment(projs)
            if 'mmd' in align_methods:
                loss_align = loss_align + pairwise_mmd_alignment(projs)

            # prototype loss (new)  增强同类样本的聚类能力，同时利用队列中的 hard negative 来增加判别力。
            # use concatenated p_all to compute prototypes per class in current batch
            # for hard negatives use concatenated queue across modalities (simple concat)
            # Build a queue for protos
            global_queue = torch.cat([queues['rgb'].get_queue()[0],
                                      queues['depth'].get_queue()[0],
                                      queues['skel'].get_queue()[0],
                                      queues['imu'].get_queue()[0]], dim=0) if (
                        queues['rgb'].get_queue()[0].size(0) > 0) else None
            # compute prototype loss on fused set of modalities by averaging proj per sample
            p_avg = (p_rgb + p_depth + p_skel + p_imu) / 4.0
            proto_loss = prototype_contrastive_loss(p_avg, labels, queue=global_queue, temperature=0.07, hard_k=hard_k)

            logits, per_modality_conf, _ = model(rgb, depth, skel, imu, mode="classification", do_completion=False)
            mask = labels >= 0
            if mask.sum() == 0: continue
            filtered_logits = logits[mask]
            filtered_label = labels[mask]
            cls_loss = criterion(filtered_logits, filtered_label)

            losses = [cls_loss, supcon_loss, loss_align, proto_loss]
            init_lambdas = [lambda_cls, lambda_supcon, lambda_align, lambda_proto]  # 根据你的偏好设置初始值

            # 打印 losses 列表里的 item
            # print([l.item() for l in losses])

            # 上一轮记录的 loss，初始化为当前值
            if 'prev_losses' not in globals():
                prev_losses = [l.item() for l in losses]
                # delta_losses 初始为 1.0，但可乘以 init_lambdas 调整比例
                delta_losses = []
                for lam in init_lambdas:
                    if lam == 0.0:
                        delta_losses.append(1.0)  # 固定loss不动态更新
                    else:
                        delta_losses.append(1.0 / lam)  # 非零loss按原逻辑

            alpha = 0.9  #平滑系数 alpha越大 lambda更新越平稳
            epsilon = 1e-6
            min_cls_lambda = 0.5  # cls 最小权重限制

            lambda_list = []
            for i, l in enumerate(losses):
                curr = l.item()
                delta = (prev_losses[i] - curr) / (prev_losses[i] + epsilon)  # 变化率
                delta_losses[i] = alpha * delta_losses[i] + (1 - alpha) * delta
                # 限制范围避免梯度爆炸
                if init_lambdas[i] == 0.0:
                    lambda_i = 0.0
                else:
                    lambda_i = init_lambdas[i] / (delta_losses[i] + epsilon)
                    lambda_i = max(min(lambda_i, 10.0), 0.01)
                # 如果是 cls，强制最小值
                if i == 0:  # cls_loss 是 losses 列表的第一个
                    lambda_i = max(lambda_i, min_cls_lambda)
                lambda_list.append(lambda_i)
                prev_losses[i] = curr  # 更新上一轮 loss

            # 可归一化
            sum_lambda = sum(lambda_list)
            lambda_list = [l / sum_lambda * len(lambda_list) for l in lambda_list]

            if if_fixedlambda:
                # 静态lambda
                loss = sum(l * w for l, w in zip(losses, init_lambdas))
            else:
                # 动态lambda
                loss = sum(l * lam for l, lam in zip(losses, lambda_list))

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_acc += (filtered_logits.argmax(dim=1) == filtered_label).sum().item()
        valid_count += filtered_label.size(0)

    avg_loss = total_loss / valid_count
    avg_acc = total_acc / valid_count
    return avg_loss, avg_acc

def validate_classification(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            skel = batch['skeleton'].to(device)
            imu = batch['imu'].to(device)
            label = batch['label'].to(device)

            logits, _, _ = model(rgb, depth, skel, imu, mode="classification", do_completion=False)
            mask = label >= 0
            if mask.sum() == 0:
                continue

            filtered_logits = logits[mask]
            filtered_label = label[mask]

            loss = criterion(filtered_logits, filtered_label)
            total_loss += loss.item() * filtered_label.size(0)
            total_acc += (filtered_logits.argmax(dim=1) == filtered_label).sum().item()
            total_count += filtered_label.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc

# ================== Main ==================
if __name__ == "__main__":
    # ===== GPU 性能与稳定性设置 =====
    torch.backends.cudnn.benchmark = True        # ✅ 自动搜索最优卷积算法，加速训练
    torch.backends.cudnn.deterministic = False   # 允许非确定性算法，更快（如想复现改为 True）
    torch.set_float32_matmul_precision('medium') # 在新版本 PyTorch 中可进一步提速 (>=2.0)
    torch.cuda.empty_cache()                     # 启动前清空缓存
    gc.collect()

    feature_root = r"D:\HAR\MMAct_preprocessed_augment\train"
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

    val_root = r"D:\HAR\MMAct_preprocessed_augment\val"
    val_dataset = PTFeatureDataset(val_root, max_frames=16)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_16,
        num_workers=8,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalHAR(num_classes=26, common_dim=128, contrastive_dim=64).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    accumulation_steps = 1

    if_pretrain = True
    # if_pretrain = False
    if if_pretrain:
        # ===== Pretrain (contrastive + prototype + align) =====
        optimizer1 = torch.optim.Adam([
            {"params": params_with_grad(model), "lr": 1e-5},  # backbone & any newly unfrozen
            {"params": model.proj_rgb.parameters(), "lr": 5e-5},
            {"params": model.proj_depth.parameters(), "lr": 5e-5},
            {"params": model.proj_skel.parameters(), "lr": 5e-5},
            {"params": model.proj_imu.parameters(), "lr": 5e-5},
            {"params": model.to_common.parameters(), "lr": 5e-5},
            {"params": model.fusion.parameters(), "lr": 5e-5}
        ], weight_decay=1e-4)

        num_pretrain_epochs = 100
        for epoch in range(num_pretrain_epochs):
            CURRENT_EPOCH = epoch
            if epoch == 10:
                for name, param in model.rgb_enc.model.layer4.named_parameters():
                    param.requires_grad = True

                optimizer1 = torch.optim.Adam([
                    {"params": params_with_grad(model), "lr": 1e-5},  # backbone & any newly unfrozen
                    {"params": model.proj_rgb.parameters(), "lr": 5e-5},
                    {"params": model.proj_depth.parameters(), "lr": 5e-5},
                    {"params": model.proj_skel.parameters(), "lr": 5e-5},
                    {"params": model.proj_imu.parameters(), "lr": 5e-5},
                    {"params": model.to_common.parameters(), "lr": 5e-5},
                    {"params": model.fusion.parameters(), "lr": 5e-5}
                ], weight_decay=1e-4)

            if epoch == 20:
                for name, param in model.rgb_enc.model.layer3.named_parameters():
                    param.requires_grad = True

                optimizer1 = torch.optim.Adam([
                    {"params": params_with_grad(model), "lr": 1e-5},  # backbone & any newly unfrozen
                    {"params": model.proj_rgb.parameters(), "lr": 5e-5},
                    {"params": model.proj_depth.parameters(), "lr": 5e-5},
                    {"params": model.proj_skel.parameters(), "lr": 5e-5},
                    {"params": model.proj_imu.parameters(), "lr": 5e-5},
                    {"params": model.to_common.parameters(), "lr": 5e-5},
                    {"params": model.fusion.parameters(), "lr": 5e-5}
                ], weight_decay=1e-4)

            loss = pretrain_contrastive_amp_supcon_with_queue(
                model, dataloader, optimizer1, device,
                lambda_cross=0.35, lambda_supcon=0.2,
                lambda_completion_recon=0.0, lambda_completion_struct=0.0,
                lambda_align=0.05, lambda_proto=0.8, if_fixedlambda=False,
                align_methods=['mean'],
                accumulation_steps=accumulation_steps,
                queue_size=2048,
                hard_k=8
            )
            print(f"[Pretrain] Epoch {epoch}: loss={loss:.4f}")
            # 保存整模型权重
            torch.save(model.state_dict(), f"pretrain_rgbimu/pretrain_epoch_{epoch}.pth")
    else:
        # ===== 加载已有 pretrain 权重 =====
        pretrain_path = "pretrain_rgbimu/pretrain_epoch_80.pth"  # 修改为你的最新 pretrain 文件
        checkpoint = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(checkpoint)


    # ================== finetune ==================
    latest_model_path = "multimodal_har_model_latest.pth"
    best_model_path = "multimodal_har_model_best.pth"
    best_val_acc = 0.0
    num_finetune_epochs = 400
    unfreezing_epoch = 20
    unfreeze_epoch = 5
    val_interval = 5

    early_stopper2 = EarlyStopping(patience=20, delta=1e-4)
    ratio_per_epoch = 1.0 / max(1, unfreezing_epoch)

    # optimizer2: different lrs per group; base: only trainable params (we will toggle requires_grad)
    optimizer2 = torch.optim.Adam([
        {"params": model.rgb_enc.parameters(), "lr": 1e-5},
        {"params": model.depth_enc.parameters(), "lr": 1e-5},
        {"params": model.skel_enc.parameters(), "lr": 1e-5},
        {"params": model.imu_enc.parameters(), "lr": 1e-5},
        {"params": model.to_common.parameters(), "lr": 3e-5},
        {"params": model.fusion.parameters(), "lr": 3e-5},
        {"params": model.fc.parameters(), "lr": 3e-5}
    ], weight_decay=5e-4)

    # optimizer2 = torch.optim.Adam([
    #     {"params": model.rgb_enc.parameters(), "lr": 5e-6},
    #     {"params": model.depth_enc.parameters(), "lr": 5e-6},
    #     {"params": model.skel_enc.parameters(), "lr": 5e-6},
    #     {"params": model.imu_enc.parameters(), "lr": 5e-6},
    #     {"params": model.to_common.parameters(), "lr": 2e-5},
    #     {"params": model.fusion.parameters(), "lr": 2e-5},
    #     {"params": model.fc.parameters(), "lr": 5e-5}
    # ], weight_decay=5e-4)

    # LR scheduler on val_loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=30, eta_min=1e-6)


    # 初始冻结 encoders
    set_encoder_trainable(model, False)
    print("🔒 冻结所有 Encoder 参数初始阶段")

    for epoch in range(num_finetune_epochs):
        CURRENT_EPOCH = epoch
        # 线性解冻
        if epoch >= unfreeze_epoch:
            unfreeze_ratio = min((epoch - unfreeze_epoch + 1) * ratio_per_epoch, 1.0)
        else:
            unfreeze_ratio = 0.0

        # 阶梯式解冻
        # if epoch < unfreeze_epoch:
        #     unfreeze_ratio = 0.0
        # elif epoch < unfreeze_epoch + 20:
        #     unfreeze_ratio = 0.25
        # elif epoch < unfreeze_epoch + 60:
        #     unfreeze_ratio = 0.5
        # elif epoch < unfreeze_epoch + 100:
        #     unfreeze_ratio = 0.75
        # else:
        #     unfreeze_ratio = 1.0

        for enc_name in ["rgb_enc", "depth_enc", "skel_enc", "imu_enc"]:
            freeze_encoder_layers_auto(model, enc_name, unfreeze_until_ratio=unfreeze_ratio)

        print(f"[Epoch {epoch}] 解冻比例: {unfreeze_ratio:.2f}")

        if epoch in [unfreeze_epoch]:
        # if epoch in [unfreeze_epoch, unfreeze_epoch + 20, unfreeze_epoch + 60, unfreeze_epoch + 100]:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[Info] Cleared GPU cache after unfreeze {unfreeze_ratio * 100:.0f}%")

        loss, acc = finetune_classification_amp(
            model, dataloader, optimizer2, criterion, device,
            accumulation_steps=accumulation_steps,
            align_methods=['mean'],
            queue_size=2048,
            hard_k=8,
            lambda_cls=2,
            lambda_supcon=0.1,
            lambda_align=0.1,
            lambda_proto=0.1,
            if_fixedlambda=False
        )
        print(f"[Finetune] Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = validate_classification(model, val_loader, criterion, device)
            print(f"[Validation] Epoch {epoch}: loss={val_loss:.4f}, acc={val_acc:.4f}")

            torch.save(model.state_dict(), latest_model_path)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"✨ 保存最佳模型: Epoch {epoch}, val_acc={val_acc:.4f}")

            scheduler.step(val_loss)

            # early_stopper2.step(val_loss)
            # if early_stopper2.early_stop:
            #     print(f"⏹️ Early stopping triggered at epoch {epoch}")
            #     break

        # 保存整模型权重
        # torch.save(model.state_dict(), f"finetune/finetune_epoch_{epoch}.pth")

    print(f"✅ 训练结束！最佳验证准确率: {best_val_acc:.4f}")

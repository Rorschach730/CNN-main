"""
PET 2D Slice Body Part Classifier (ResNet-18)
=============================================
解决 Z-slice 阈值分类不准确的问题（老人/儿童身高差异、体位差异）。

方案:
  1. 用 v4 格式数据训练 ResNet-18 三分类器 (Brain/Chest/Abdomen)
  2. 对 legacy 2 通道数据自动打标并升级为 v4 3 通道格式
  3. 人体部位无关身高，基于 2D PET 切片解剖结构判断

v4 格式说明:
  tensor[3, H, W]
    [0] = body_part_id (int, 0=Brain, 1=Chest, 2=Abdomen)
    [1] = condition (低剂量)
    [2] = target (全剂量)

用法 (所有参数已硬编码，直接跑):
  python trido_ud/body_part_classifier.py      # 自动训练+打标
  python trido_ud/body_part_classifier.py train  # 仅训练
  python trido_ud/body_part_classifier.py batch  # 仅打标 (需已有 ckpt)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import numpy as np
from tqdm import tqdm

# ═══════════════════════════════════════════════════
# 🔧 硬编码默认参数 (按你的环境修改)
# ═══════════════════════════════════════════════════

CONFIG = {
    # 训练数据 (v4 格式, 含 body_part 标签)
    'train_data': 'I:/processed_data_trido/train',

    # 待打标的 legacy 数据 (2 通道, 无 body_part)
    'legacy_data': 'I:/processed_data_trido/test',

    # 输出
    'ckpt_path': './body_part_ckpt.pth',

    # 训练参数
    'epochs': 30,
    'batch_size': 64,
    'lr': 1e-3,
    'img_size': 256,
}


# ═══════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════

class BodyPartDataset(Dataset):
    """
    从 v4 格式 .pt 文件加载: tensor[3,H,W] = [body_part, condition, target]
    使用 condition 作为分类输入。
    """

    def __init__(self, data_dir: str, img_size: int = 256):
        self.samples = []
        self.img_size = img_size

        patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))]

        for p_dir in patient_dirs:
            for f in os.listdir(p_dir):
                if f.endswith('.pt'):
                    self.samples.append(os.path.join(p_dir, f))

        print(f"[*] BodyPartDataset: {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor = torch.load(self.samples[idx], map_location='cpu', weights_only=True)

        if tensor.ndim == 3 and tensor.shape[0] == 3:
            body_part = int(tensor[0, 0, 0].item())
            condition = tensor[1:2, :, :].float()
        else:
            body_part = -1
            condition = tensor[0:1, :, :].float()

        _, H, W = condition.shape
        if H != self.img_size or W != self.img_size:
            condition = F.interpolate(
                condition.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False).squeeze(0)

        condition = condition.repeat(3, 1, 1)
        vmax = condition.max()
        if vmax > 0:
            condition = condition / vmax

        return condition, body_part


# ═══════════════════════════════════════════════════
# Model: ResNet-18 三分类器
# ═══════════════════════════════════════════════════

class BodyPartClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ═══════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════

def train():
    cfg = CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = BodyPartDataset(cfg['train_data'], img_size=cfg['img_size'])
    valid_idx = [i for i in range(len(dataset)) if dataset[i][1] >= 0]
    if not valid_idx:
        print(f"[!] 训练数据中没有 v4 格式样本 (需要 tensor[0] = body_part 标签)")
        print(f"    请先确认 {cfg['train_data']} 目录下的 .pt 文件是 3 通道 v4 格式")
        sys.exit(1)

    dataset = Subset(dataset, valid_idx)
    print(f"[*] 有效训练样本: {len(dataset)}")

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=4)

    model = BodyPartClassifier(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total

        print(f"  Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg['ckpt_path'])
            print(f"  → 保存最佳模型 (Acc={acc:.4f})")

    print(f"\n[√] 训练完成. 最佳验证准确率: {best_acc:.4f}")
    print(f"    模型已保存: {cfg['ckpt_path']}")
    return best_acc


# ═══════════════════════════════════════════════════
# Single Prediction
# ═══════════════════════════════════════════════════

@torch.no_grad()
def predict_single(model, pt_path: str, device: torch.device) -> int:
    """对单个 .pt 文件预测 body_part (0/1/2)"""
    tensor = torch.load(pt_path, map_location='cpu', weights_only=True)

    if tensor.ndim == 3 and tensor.shape[0] >= 2:
        cond_idx = 1 if tensor.shape[0] == 3 else 0
        condition = tensor[cond_idx:cond_idx+1, :, :].float()
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    _, H, W = condition.shape
    if H != CONFIG['img_size'] or W != CONFIG['img_size']:
        condition = F.interpolate(
            condition.unsqueeze(0), size=(CONFIG['img_size'], CONFIG['img_size']),
            mode='bilinear', align_corners=False).squeeze(0)
    condition = condition.repeat(3, 1, 1)
    vmax = condition.max()
    if vmax > 0:
        condition = condition / vmax

    model.eval()
    logits = model(condition.unsqueeze(0).to(device))
    return logits.argmax(dim=1).item()


# ═══════════════════════════════════════════════════
# Batch Relabel (含写回 .pt 文件)
# ═══════════════════════════════════════════════════

def batch_relabel(write_back: bool = True):
    """
    批量打标，默认写回 .pt 文件。

    逻辑:
      - 2 通道 legacy:  [condition, target] → [body_part, condition, target]
      - 3 通道 v4:      [旧body_part, condition, target] → [新body_part, condition, target]
      - 写回同一个文件 (原地覆盖)

    Args:
        write_back: False 时仅统计不修改文件
    """
    cfg = CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BodyPartClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(cfg['ckpt_path'], map_location=device))
    model.eval()

    BP_NAMES = {0: 'Brain', 1: 'Chest', 2: 'Abdomen'}
    stats = {0: 0, 1: 0, 2: 0}
    converted = 0   # 2→3 通道升级数
    corrected = 0   # 原 body_part 被修正数
    errors = 0
    total = 0

    data_path = cfg['legacy_data']
    patient_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, d))]

    if not patient_dirs:
        print(f"[!] {data_path} 下没有找到患者子目录")
        sys.exit(1)

    action = "打标并写回文件" if write_back else "统计 (不修改文件)"
    print(f"[*] 模式: {action}")
    print(f"[*] 数据: {data_path} ({len(patient_dirs)} 个患者目录)\n")

    for p_dir in tqdm(patient_dirs, desc="Relabeling"):
        pt_files = [f for f in os.listdir(p_dir) if f.endswith('.pt')]
        for f in pt_files:
            fpath = os.path.join(p_dir, f)
            try:
                bp = predict_single(model, fpath, device)
                stats[bp] += 1
                total += 1

                if write_back:
                    tensor = torch.load(fpath, map_location='cpu', weights_only=True)
                    n_ch = tensor.shape[0]

                    if n_ch == 2:
                        # Legacy: 升级为 v4 格式
                        new_tensor = torch.zeros(3, tensor.shape[1], tensor.shape[2])
                        new_tensor[0, 0, 0] = float(bp)  # body_part 存入第一维
                        new_tensor[1:2, :, :] = tensor[0:1, :, :]  # condition
                        new_tensor[2:3, :, :] = tensor[1:2, :, :]  # target
                        torch.save(new_tensor, fpath)
                        converted += 1

                    elif n_ch == 3:
                        # v4: 更新 body_part
                        old_bp = int(tensor[0, 0, 0].item())
                        if old_bp != bp:
                            tensor[0, :, :] = float(bp)  # 整行写入 body_part 值
                            torch.save(tensor, fpath)
                            corrected += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [!] Skip {fpath}: {e}")

    print(f"\n{'='*50}")
    print(f"  处理完成: {total} 个切片")
    for k, v in stats.items():
        print(f"    {BP_NAMES[k]:<10} {v:>6}  ({v/total*100:.1f}%)")
    if write_back:
        print(f"    2→3 通道升级: {converted}")
        print(f"    body_part 修正: {corrected}")
    if errors:
        print(f"    跳过/错误: {errors}")
    print(f"{'='*50}")


# ═══════════════════════════════════════════════════
# Main: 自动流水线 (训练 → 打标)
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('PET Body Part Classifier')
    parser.add_argument('mode', nargs='?', default='auto',
                        choices=['auto', 'train', 'batch', 'stats'],
                        help='auto=训练+打标 | train=仅训练 | batch=训练+写回 | stats=仅统计')
    parser.add_argument('--no-write', action='store_true',
                        help='batch 模式不写回文件 (仅统计)')
    args = parser.parse_args()

    if args.mode == 'train':
        train()

    elif args.mode == 'stats':
        batch_relabel(write_back=False)

    elif args.mode == 'batch':
        batch_relabel(write_back=not args.no_write)

    elif args.mode == 'auto':
        # 自动流水线: 训 → 打
        print("╔══════════════════════════════════════════╗")
        print("║  Body Part Classifier — 自动流水线       ║")
        print("╚══════════════════════════════════════════╝\n")
        train()
        print()
        batch_relabel(write_back=True)

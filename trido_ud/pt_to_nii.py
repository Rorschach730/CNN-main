"""
PT to NIfTI Converter & Dataset Loader
=======================================
解决 exFAT 1MB 簇存储浪费问题: 上万个小 .pt 文件 → 按患者/剂量合并为 .nii.gz 卷。

存储对比:
  Before: patient/xxx_D2_Z0001.pt, _Z0002.pt, ... (~400 个/患者/剂量)
          → 每个 300KB 文件占 1MB (exFAT 1MB 簇) → 浪费 ~70%
  After:  patient_D2.nii.gz (一个文件, gzip 压缩)
          → 300KB × 400 slices ≈ 120MB → gzip ~30MB → 零簇浪费

NIfTI shape: (H, W, n_slices, C)
  v4 (3通道):  [body_part, condition, target]
  legacy (2通道): [condition, target]

训练适配: NiftiSliceDataset 完全兼容 TriDoPETDataset 接口

用法 (直接跑, 参数已硬编码):
  python trido_ud/pt_to_nii.py
"""

import os
import re
from collections import defaultdict

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm


# ═══════════════════════════════════════════════════
# 🔧 硬编码默认参数 (按你的环境修改)
# ═══════════════════════════════════════════════════

CONFIG = {
    'input_dir': 'I:/processed_data_trido',        # 源 .pt 数据
    'output_dir': 'I:/processed_data_trido_nii',   # 输出 .nii.gz
    'img_size': 256,
}


# ═══════════════════════════════════════════════════
# 文件名解析
# ═══════════════════════════════════════════════════

def parse_filename(filename: str):
    """
    从文件名提取元信息。
    例: 20221021_9_20221021_165244_D10_Z0428.pt
    → (patient='20221021_9', dose='10', z=428)
    """
    base = filename.replace('.pt', '')
    # Dose: _D{int}_
    dose_match = re.search(r'_D(\d+)_', base)
    dose = dose_match.group(1) if dose_match else 'unknown'
    # Z-slice: _Z{int}
    z_match = re.search(r'_Z(\d+)', base)
    z = int(z_match.group(1)) if z_match else 0
    # Patient: everything before the date-like part
    # Pattern: {patient_id}_{YYYYMMDD}_{HHMMSS}_D{dose}_Z{slice}
    parts = base.split('_')
    patient = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]

    return patient, dose, z


def convert_pt_to_nii():
    """
    扫描 CONFIG['input_dir'], 按 (patient, dose) 分组, 合并为 .nii.gz。
    """
    input_dir = CONFIG['input_dir']
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # ── 扫描所有 .pt 文件 ──
    print(f"[*] Scanning {input_dir}...")
    groups = defaultdict(list)  # (patient, dose) → [(z, filepath)]

    patient_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, d))]

    for p_dir in tqdm(patient_dirs, desc="Scanning"):
        for f in os.listdir(p_dir):
            if not f.endswith('.pt'):
                continue
            fpath = os.path.join(p_dir, f)
            patient, dose, z = parse_filename(f)
            groups[(patient, dose)].append((z, fpath))

    print(f"[*] Found {sum(len(v) for v in groups.values())} slices "
          f"in {len(groups)} volumes")

    # ── 逐卷合并 ──
    for (patient, dose), slices in tqdm(groups.items(), desc="Converting"):
        # 按 Z 排序
        slices.sort(key=lambda x: x[0])

        # 读取第一张确定通道数
        first = torch.load(slices[0][1], map_location='cpu', weights_only=True)
        n_channels = first.shape[0] if first.ndim == 3 else 1
        H, W = first.shape[-2:] if first.ndim >= 2 else (256, 256)
        n_slices = len(slices)

        # 分配数组
        volume = np.zeros((n_slices, n_channels, H, W), dtype=np.float32)

        for i, (z, fpath) in enumerate(slices):
            tensor = torch.load(fpath, map_location='cpu', weights_only=True).numpy()
            volume[i] = tensor

        # NIfTI expects (H, W, n_slices) or (H, W, n_slices, C)
        # 转置: (S, C, H, W) → (H, W, S, C)
        volume_nii = np.transpose(volume, (2, 3, 0, 1))

        # 保存
        out_name = f"{patient}_D{dose}.nii.gz"
        out_path = os.path.join(output_dir, out_name)
        img = nib.Nifti1Image(volume_nii, affine=np.eye(4))
        nib.save(img, out_path)

    # ── 统计 ──
    original_size = 0
    for (_, _), slices in groups.items():
        for _, fpath in slices:
            original_size += os.path.getsize(fpath)
    final_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith('.nii.gz')
    )
    print(f"\n[√] Conversion done.")
    print(f"  Original: {original_size / 1e9:.2f} GB")
    print(f"  NIfTI:    {final_size / 1e9:.2f} GB")
    print(f"  Ratio:    {final_size/original_size*100:.1f}%")

    # 磁盘空间浪费估算
    cluster_size = 1024 * 1024  # 1MB
    total_files = sum(len(v) for v in groups.values())
    wasted = total_files * cluster_size - original_size
    print(f"  exFAT 簇浪费估算: {wasted / 1e9:.2f} GB ({total_files} files × 1MB)")


# ═══════════════════════════════════════════════════════════════
# NiftiSliceDataset: 训练时从 .nii.gz 读取
# ═══════════════════════════════════════════════════════════════

class NiftiSliceDataset:
    """
    替代 TriDoPETDataset: 从 .nii.gz 文件中按 Z 索引读取单切片。

    目录结构:
      data_dir/
        patient_D{dose}.nii.gz  (每个文件 = 一个 volume)

    __getitem__ 返回:
      (target, condition, body_part_tensor)  # 兼容现有 pipeline
    """

    def __init__(self, data_dir: str, img_size: int = 256,
                 virtual_epoch_ratio: float = 1.0):
        self.data_dir = data_dir
        self.img_size = img_size
        self.virtual_epoch_ratio = virtual_epoch_ratio

        # 扫描 .nii.gz 文件, 建立 (file_idx, z_idx) 索引
        self.index = []
        self.nii_files = []

        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.nii.gz'):
                fpath = os.path.join(data_dir, f)
                # 快速读取 header 获取维度
                img = nib.load(fpath)
                shape = img.shape  # (H, W, n_slices, C)
                if len(shape) == 4:
                    n_slices = shape[2]
                elif len(shape) == 3:
                    n_slices = shape[2]
                else:
                    continue

                file_idx = len(self.nii_files)
                self.nii_files.append(fpath)
                for z in range(n_slices):
                    self.index.append((file_idx, z))

                if self.virtual_epoch_ratio < 1.0:
                    import random
                    random.shuffle(self.index)
                    n_virtual = max(1, int(len(self.index) * self.virtual_epoch_ratio))
                    self.index = self.index[:n_virtual]

        print(f"[*] NiftiSliceDataset: {len(self.nii_files)} volumes, "
              f"{len(self.index)} total slices")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, z_idx = self.index[idx]
        img = nib.load(self.nii_files[file_idx])
        data = img.get_fdata()  # (H, W, S, C)

        if data.ndim == 3:
            # Legacy: (H, W, S) → 假设单通道 condition
            condition = torch.from_numpy(data[:, :, z_idx]).float().unsqueeze(0)
            target = condition.clone()  # No target available
            body_part = torch.tensor(0, dtype=torch.long)
        else:
            # v4: (H, W, S, C), C = 3: [body_part, condition, target]
            slice_data = data[:, :, z_idx, :]  # (H, W, C)
            slice_tensor = torch.from_numpy(slice_data).float()

            if slice_tensor.shape[-1] == 3:
                body_part = int(slice_tensor[0, 0, 0].item())  # body_part 是常数
                condition = slice_tensor[:, :, 1].unsqueeze(0)  # (1, H, W)
                target = slice_tensor[:, :, 2].unsqueeze(0)
            elif slice_tensor.shape[-1] == 2:
                body_part = torch.tensor(0, dtype=torch.long)
                condition = slice_tensor[:, :, 0].unsqueeze(0)
                target = slice_tensor[:, :, 1].unsqueeze(0)
            else:
                condition = slice_tensor[:, :, 0].unsqueeze(0)
                target = condition.clone()
                body_part = torch.tensor(0, dtype=torch.long)

        # Resize
        if self.img_size is not None:
            _, H, W = condition.shape
            if H != self.img_size or W != self.img_size:
                condition = torch.nn.functional.interpolate(
                    condition.unsqueeze(0), size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False).squeeze(0)
                target = torch.nn.functional.interpolate(
                    target.unsqueeze(0), size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False).squeeze(0)

        body_part_tensor = torch.tensor(body_part, dtype=torch.long)
        return target, condition, body_part_tensor


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════╗")
    print("║  PT → NIfTI 转换器                       ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  输入: {CONFIG['input_dir']}")
    print(f"  输出: {CONFIG['output_dir']}")
    print()
    convert_pt_to_nii()

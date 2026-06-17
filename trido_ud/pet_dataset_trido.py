"""
TriDo-CNN Dataset Loader
=========================
PET denoising dataset for triple-domain training.

Data format:
  1. v4 .pt files (from udpet_cleaner_trido.py): [3, H, W] tensors
     with [0]=body_part, [1]=condition (low-dose), [2]=target (full-dose)
  2. Legacy .pt files: [2, H, W] tensors with condition + target
     (body_part inferred from Z-slice position or body_part_map fallback)

The dataset returns (target, condition, body_part) tuples compatible
with both image-only and triple-domain training.

Backward compatible with the _ud dataset format.
Exports: TriDoPETDataset (primary), PETDatasetTrido (alias),
         PETDenoisingDataset (2-tuple wrapper for legacy code).
"""

import os
import re
import random
import torch
from torch.utils.data import Dataset


class TriDoPETDataset(Dataset):
    """
    PET Denoising Dataset for TriDo-CNN training with Virtual Epoch support.
    """

    def __init__(self, data_dir, img_size=256, body_part_map=None, virtual_epoch_ratio=0.10, seed=42):
        """
        Args:
            data_dir: Path to data directory
            img_size: Expected image size (None = no resize)
            body_part_map: Legacy fallback mapping
            virtual_epoch_ratio: Fraction of data to use per "epoch" (e.g., 0.05 means 1/20th)
            seed: Random seed for deterministic shuffling
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.body_part_map = body_part_map
        self.samples = []
        
        # Virtual Epoch parameters
        self.virtual_epoch_ratio = virtual_epoch_ratio
        self.seed = seed
        self.current_epoch_samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据路径不存在: {data_dir}")

        print(f"[*] 正在扫描 {data_dir} 构建三域训练样本索引...")

        patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))]

        for p_dir in patient_dirs:
            pt_files = [os.path.join(p_dir, f) for f in os.listdir(p_dir)
                        if f.endswith('.pt')]
            self.samples.extend(pt_files)

        if len(self.samples) == 0:
            print(f"[!] 警告：在 {data_dir} 中未发现有效的 .pt 样本文件。")
        else:
            self.total_samples = len(self.samples)
            self.virtual_length = max(1, int(self.total_samples * self.virtual_epoch_ratio))
            print(f"[*] 索引构建完成，共锁定 {self.total_samples} 个三域物理样本。")
            print(f"[*] 开启【虚拟短 Epoch】模式: 每个 Epoch 随机抽样 {self.virtual_length} 个样本 (占比 {self.virtual_epoch_ratio*100:.1f}%)。")
            
            # Initialize the first virtual epoch
            self._shuffle_and_sample()

    def _shuffle_and_sample(self):
        """Randomly select a subset of samples for the current virtual epoch."""
        # Use a local Random instance to avoid messing with global seeds
        rng = random.Random(self.seed)
        
        # We increment the seed for the next epoch so we get different data each time
        self.seed += 1 
        
        # Create a copy to shuffle
        shuffled_samples = self.samples.copy()
        rng.shuffle(shuffled_samples)
        
        # Take the subset
        self.current_epoch_samples = shuffled_samples[:self.virtual_length]

    def set_epoch(self, epoch):
        """
        Called by the training loop at the start of each epoch to resample data.
        """
        self._shuffle_and_sample()
        print(f"\n[*] Dataset resampled for Epoch {epoch}: Using {self.virtual_length} random samples.")

    def __len__(self):
        return len(self.current_epoch_samples)

    @staticmethod
    def _parse_z_slice(filename):
        match = re.search(r'_Z(\d{4})', filename)
        if match:
            return int(match.group(1))
        return None

    def _legacy_body_part(self, filename):
        if isinstance(self.body_part_map, dict):
            match = re.search(r'(P\d+)', filename)
            if match:
                patient_id = match.group(1)
                for prefix, label in sorted(self.body_part_map.items(), key=lambda x: -len(x[0])):
                    if patient_id.startswith(prefix):
                        return label

        if callable(self.body_part_map):
            return self.body_part_map(filename)

        z = self._parse_z_slice(filename)
        if z is not None:
            if z < 50: return 0
            elif z < 120: return 1
            else: return 2
        return 0

    def __getitem__(self, idx):
        tries = getattr(self, '_getitem_tries', 0)
        if tries > len(self.current_epoch_samples):
            self._getitem_tries = 0
            raise RuntimeError("All samples in current virtual epoch failed to load.")

        file_path = self.current_epoch_samples[idx % len(self.current_epoch_samples)]
        filename = os.path.basename(file_path)

        try:
            tensor_pair = torch.load(file_path, map_location='cpu', weights_only=True)

            if tensor_pair.ndim != 3 or tensor_pair.shape[0] not in (2, 3):
                raise ValueError(f"Unexpected tensor shape: {tensor_pair.shape}")

            if tensor_pair.shape[0] == 3:
                body_part = int(tensor_pair[0, 0, 0].item())
                condition_tensor = tensor_pair[1:2, :, :].float()
                target_tensor    = tensor_pair[2:3, :, :].float()
            else:
                condition_tensor = tensor_pair[0:1, :, :].float()
                target_tensor    = tensor_pair[1:2, :, :].float()
                body_part = self._legacy_body_part(filename)

            body_part_tensor = torch.tensor(body_part, dtype=torch.long)

            if self.img_size is not None:
                _, H, W = target_tensor.shape
                if H != self.img_size or W != self.img_size:
                    target_tensor = torch.nn.functional.interpolate(
                        target_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                        mode='bilinear', align_corners=False).squeeze(0)
                    condition_tensor = torch.nn.functional.interpolate(
                        condition_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                        mode='bilinear', align_corners=False).squeeze(0)

            self._getitem_tries = 0
            return target_tensor, condition_tensor, body_part_tensor

        except Exception as e:
            self._getitem_tries = tries + 1
            new_idx = (idx + 1) % len(self.current_epoch_samples)
            return self.__getitem__(new_idx)

# Alias
PETDatasetTrido = TriDoPETDataset

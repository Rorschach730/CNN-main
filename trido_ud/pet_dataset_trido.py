"""
TriDo-JiT Dataset Loader
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
import torch
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════
#  Z-slice → body-part 对照表（保留供外部参考）
#  Brain:  Z0000–Z0049  |  Chest:  Z0050–Z0119  |  Abdomen: Z0120–Z0200
#  ⚠️ v4 .pt 文件自带 body_part 通道 [0,:,:]，此表仅用于 legacy 2 通道 fallback。
# ═══════════════════════════════════════════════════════════════════


class TriDoPETDataset(Dataset):
    """
    PET Denoising Dataset for TriDo-JiT training.

    Directory structure (compatible with _ud):
    root/
    ├── train/
    │   ├── P0001/
    │   │   ├── P0001_D10_Z0000.pt  (with body part label)
    │   │   └── ...
    │   └── P0002/
    └── test/

    Each .pt file contains:
      - v4 format [3, H, W]: [0]=body_part, [1]=condition (low-dose), [2]=target (full-dose)
      - legacy format [2, H, W]: [0]=condition, [1]=target (body_part from body_part_map fallback)

    Body part categories (for embedding in model):
      0 = brain (脑部)
      1 = chest (胸部)
      2 = abdomen (腹部)
    """

    def __init__(self, data_dir, img_size=256, body_part_map=None):
        """
        Args:
            data_dir: Path to data directory
            img_size: Expected image size (None = no resize)
            body_part_map: dict mapping patient_prefix → body_part (0/1/2)
                           or callable(filename) → body_part.
                           Only used as fallback when .pt files have 2 channels
                           (legacy format).  v4 3-channel .pt files read body_part
                           directly from tensor.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.body_part_map = body_part_map
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据路径不存在: {data_dir}")

        print(f"[*] 正在扫描 {data_dir} 构建三域训练样本索引...")

        # Scan patient subdirectories
        patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))]

        for p_dir in patient_dirs:
            pt_files = [os.path.join(p_dir, f) for f in os.listdir(p_dir)
                        if f.endswith('.pt')]
            self.samples.extend(pt_files)

        if len(self.samples) == 0:
            print(f"[!] 警告：在 {data_dir} 中未发现有效的 .pt 样本文件。")
        else:
            print(f"[*] 索引构建完成，共锁定 {len(self.samples)} 个三域物理样本。")

    def __len__(self):
        return len(self.samples)

    # ── Static helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_z_slice(filename):
        """Extract Z-slice layer number from filename.

        Example: 'P0001_D10_Z0050.pt' → 50
        Returns None if no Z-slice number found.
        """
        match = re.search(r'_Z(\d{4})', filename)
        if match:
            return int(match.group(1))
        return None

    # ── Legacy body-part fallback (for 2-channel .pt files only) ─────

    def _legacy_body_part(self, filename):
        """Fallback body_part for legacy 2-channel .pt files.

        Priority:
          1. body_part_map dict: match patient prefix → 0/1/2
             (P[0-9]+ regex only matches old P0001-style IDs; new IDs use cleaner-baked body_part)
          2. body_part_map callable: invoke fn(filename) → 0/1/2
          3. Z-slice heuristics: Z0000-0049→0(brain), Z0050-0119→1(chest), Z0120+→2(abdomen)
        """
        # Priority 1: body_part_map dict
        if isinstance(self.body_part_map, dict):
            match = re.search(r'(P\d+)', filename)
            if match:
                patient_id = match.group(1)
                for prefix, label in sorted(self.body_part_map.items(), key=lambda x: -len(x[0])):
                    if patient_id.startswith(prefix):
                        return label

        # Priority 2: body_part_map callable
        if callable(self.body_part_map):
            return self.body_part_map(filename)

        # Priority 3: Z-slice heuristic
        z = self._parse_z_slice(filename)
        if z is not None:
            if z < 50:
                return 0   # brain
            elif z < 120:
                return 1   # chest
            else:
                return 2   # abdomen
        return 0  # default

    # ── Main item access ──────────────────────────────────────────

    def __getitem__(self, idx):
        """
        Returns (target_tensor, condition_tensor, body_part_tensor).

        Reads from v4 [3, H, W] format (preferred) or legacy [2, H, W]
        format with body_part fallback.

        On loading errors, skips forward to the next sample with a
        recursion guard to prevent infinite loops on fully corrupted
        datasets.
        """
        # Recursion guard: track how many consecutive failures we tolerate
        tries = getattr(self, '_getitem_tries', 0)
        if tries > len(self.samples):
            # All samples have been tried — raise to let DataLoader skip
            self._getitem_tries = 0
            raise RuntimeError(
                f"All {len(self.samples)} samples failed to load. "
                f"Check data integrity in {self.data_dir}"
            )

        file_path = self.samples[idx % len(self.samples)]
        filename = os.path.basename(file_path)

        try:
            # Load tensor pair
            tensor_pair = torch.load(file_path, map_location='cpu', weights_only=True)

            # Validate shape: expect [2, H, W] or [3, H, W]
            if tensor_pair.ndim != 3 or tensor_pair.shape[0] not in (2, 3):
                raise ValueError(f"Unexpected tensor shape: {tensor_pair.shape}")

            if tensor_pair.shape[0] == 3:
                # ── v4 format [3, H, W] — body_part baked in ──
                body_part = int(tensor_pair[0, 0, 0].item())
                condition_tensor = tensor_pair[1:2, :, :].float()  # (1, H, W)
                target_tensor    = tensor_pair[2:3, :, :].float()  # (1, H, W)
            else:
                # ── Legacy format [2, H, W] — body_part from fallback ──
                condition_tensor = tensor_pair[0:1, :, :].float()  # (1, H, W)
                target_tensor    = tensor_pair[1:2, :, :].float()  # (1, H, W)
                body_part = self._legacy_body_part(filename)

            # Convert to LongTensor for embedding lookup
            body_part_tensor = torch.tensor(body_part, dtype=torch.long)

            # Resize if needed
            if self.img_size is not None:
                _, H, W = target_tensor.shape
                if H != self.img_size or W != self.img_size:
                    target_tensor = torch.nn.functional.interpolate(
                        target_tensor.unsqueeze(0),
                        size=(self.img_size, self.img_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    condition_tensor = torch.nn.functional.interpolate(
                        condition_tensor.unsqueeze(0),
                        size=(self.img_size, self.img_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

            # Reset recursion guard on success
            self._getitem_tries = 0

            return target_tensor, condition_tensor, body_part_tensor

        except Exception as e:
            # Skip corrupted samples — advance with guard
            self._getitem_tries = tries + 1
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)


# ═══════════════════════════════════════════════════════════════════
#  Backward-compatible aliases and wrappers
# ═══════════════════════════════════════════════════════════════════

# Alias for code expecting the old class name
PETDatasetTrido = TriDoPETDataset


class PETDenoisingDataset(Dataset):
    """
    Backward-compatible dataset wrapper.

    Auto-detects data format:
      - .pt files (from udpet_cleaner.py): delegates to TriDoPETDataset
      - .npy files (legacy OSEM pipeline): loads 3D volumes, returns 2D slices

    Always returns (target, condition) 2-tuples.

    Also exposes .samples as a list of (file_path, z_slice) tuples
    for compatibility with test_evaluation.py and similar scripts.

    Usage (drop-in replacement):
        from trido_ud.pet_dataset_trido import PETDenoisingDataset
        ds = PETDenoisingDataset('./processed_data_udpet/test', img_size=128)
        target, condition = ds[0]
    """

    def __init__(self, data_dir, img_size=128, is_train=True):
        self._img_size = img_size
        self._data_dir = data_dir
        self._mode = None  # 'pt' or 'npy'

        # ── Probe: .pt files first (UDPET / TriDo pipeline) ──
        pt_files = []
        if os.path.isdir(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.pt'):
                        pt_files.append(os.path.join(root, f))
                        if len(pt_files) > 100:  # enough to confirm
                            break
                if len(pt_files) > 100:
                    break

        if pt_files:
            # .pt format detected — delegate to TriDoPETDataset
            self._mode = 'pt'
            self._inner = TriDoPETDataset(data_dir, img_size=img_size)
            self.samples = []
            for fp in self._inner.samples:
                z = TriDoPETDataset._parse_z_slice(os.path.basename(fp))
                self.samples.append((fp, z if z is not None else 0))
            return

        # ── Fallback: .npy files (legacy OSEM pipeline) ──
        import glob
        import numpy as np

        self._mode = 'npy'
        self._npy_file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
        if len(self._npy_file_paths) == 0:
            print(f"Warning: No .pt or .npy files found in {data_dir}")

        self._npy_index_map = []   # [(file_idx, slice_idx), ...]
        self._npy_data_cache = []
        self.samples = []          # [(file_path, z_slice), ...]

        print(f"Pre-loading data from {data_dir}...")
        for i, fp in enumerate(self._npy_file_paths):
            try:
                data = np.load(fp, allow_pickle=True).item()
                self._npy_data_cache.append({
                    'input': data['input'].astype(np.float32),
                    'target': data['target'].astype(np.float32)
                })
                depth = data['input'].shape[0]
                for d in range(depth):
                    self._npy_index_map.append((i, d))
                    self.samples.append((fp, d))
            except Exception as e:
                print(f"Error loading {fp}: {e}")

        print(f"Loaded {len(self._npy_file_paths)} volumes, "
              f"Total slices: {len(self._npy_index_map)}")

    def __len__(self):
        if self._mode == 'pt':
            return len(self._inner)
        return len(self._npy_index_map)

    def __getitem__(self, idx):
        if self._mode == 'pt':
            target, condition, _body_part = self._inner[idx]
            return target, condition

        # ── .npy mode ──
        import numpy as np

        file_idx, slice_idx = self._npy_index_map[idx]
        data = self._npy_data_cache[file_idx]

        img_input = data['input'][slice_idx]   # Noisy
        img_target = data['target'][slice_idx]  # Clean

        # Robust Min-Max Normalization to [-1, 1]
        v_min = min(img_input.min(), img_target.min())
        v_max = max(img_input.max(), img_target.max())
        scale = v_max - v_min
        if scale < 1e-6:
            scale = 1.0

        img_input = (img_input - v_min) / scale * 2.0 - 1.0
        img_target = (img_target - v_min) / scale * 2.0 - 1.0

        img_input = np.clip(img_input, -1.0, 1.0)
        img_target = np.clip(img_target, -1.0, 1.0)

        img_input = torch.from_numpy(img_input).unsqueeze(0)
        img_target = torch.from_numpy(img_target).unsqueeze(0)

        return img_target, img_input

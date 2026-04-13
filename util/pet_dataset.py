import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class PETDenoisingDataset(Dataset):
    def __init__(self, root_dir, img_size=128, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.file_paths = glob.glob(os.path.join(root_dir, "*.npy"))

        if len(self.file_paths) == 0:
            print(f"Warning: No .npy files found in {root_dir}")

        self.index_map = []
        self.data_cache = []

        print(f"Pre-loading data from {root_dir}...")
        for i, fp in enumerate(self.file_paths):
            try:
                # 此时硬盘里的 input 已经是带条纹伪影的物理仿真数据
                data = np.load(fp, allow_pickle=True).item()

                # 存入内存
                self.data_cache.append({
                    'input': data['input'].astype(np.float32),
                    'target': data['target'].astype(np.float32)
                })

                depth = data['input'].shape[0]
                for d in range(depth):
                    self.index_map.append((i, d))
            except Exception as e:
                print(f"Error loading {fp}: {e}")

        print(f"Loaded {len(self.file_paths)} volumes, Total slices: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.index_map[idx]
        data = self.data_cache[file_idx]

        img_input = data['input'][slice_idx]  # Noisy (Sinogram Simulated)
        img_target = data['target'][slice_idx]  # Clean

        # -----------------------------------------------------------
        # [归一化逻辑] Robust Min-Max Normalization
        # 目的：将物理仿真后的任意数值范围映射到 Diffsuion 喜欢的 [-1, 1]
        # 关键点：使用 Input 和 Target 的并集范围，保留两者之间的相对强度差异（噪声幅度）
        # -----------------------------------------------------------

        v_min = min(img_input.min(), img_target.min())
        v_max = max(img_input.max(), img_target.max())

        scale = v_max - v_min
        if scale < 1e-6: scale = 1.0

        # 1. 归一化到 [0, 1]
        img_input = (img_input - v_min) / scale
        img_target = (img_target - v_min) / scale

        # 2. 映射到 [-1, 1]
        img_input = img_input * 2.0 - 1.0
        img_target = img_target * 2.0 - 1.0

        # 3. 安全截断
        img_input = np.clip(img_input, -1.0, 1.0)
        img_target = np.clip(img_target, -1.0, 1.0)

        # 转 Tensor [1, H, W]
        img_input = torch.from_numpy(img_input).unsqueeze(0)
        img_target = torch.from_numpy(img_target).unsqueeze(0)

        return img_target, img_input
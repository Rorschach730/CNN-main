import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PETDenoisingDataset(Dataset):
    def __init__(self, data_dir, img_size=128):
        self.data_dir = data_dir
        self.img_size = img_size
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

        print(f"[*] 正在扫描 {data_dir} 构建 2D 张量切片索引...")
        self.samples = []
        for f in self.files:
            try:
                # 仅在初始化时快速读取并建立 Z 轴映射索引
                data = np.load(f, allow_pickle=True).item()
                d = data['input'].shape[0]
                for z in range(d):
                    self.samples.append((f, z))
            except Exception as e:
                pass
        print(f"[*] 索引构建完成，共锁定 {len(self.samples)} 个 2D 物理样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, z = self.samples[idx]

        # 依赖操作系统 Page Cache 实现底层高速复用读取
        data = np.load(file_path, allow_pickle=True).item()
        noisy_vol = data['input']
        clean_vol = data['target']

        # [降维切除]: 彻底抛弃 Z 轴相干性，只提取绝对中心层
        # 保持维度为 (1, H, W)
        condition_img = noisy_vol[z:z+1, :, :]
        target_img = clean_vol[z:z+1, :, :]

        # 转换为高精度张量
        condition_tensor = torch.from_numpy(condition_img).float()
        target_tensor = torch.from_numpy(target_img).float()

        return target_tensor, condition_tensor
import os
import torch
import re
from torch.utils.data import Dataset


class PETDenoisingDataset(Dataset):
    """
    针对 UDPET 2D 混合剂量（Mixed-Dose）张量数据的深度学习训练数据集。
    数据结构要求：
    root/
    ├── train/
    │   ├── P0001/
    │   │   ├── P0001_D10_Z0000.pt  (D10 代表 1/10 剂量)
    │   │   ├── P0001_D4_Z0001.pt   (D4 代表 1/4 剂量)
    │   │   └── ...
    │   └── P0002/
    └── test/
    """

    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据路径不存在: {data_dir}")

        print(f"[*] 正在扫描 {data_dir} 构建混合剂量样本索引...")

        # 遍历病人子文件夹
        patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))]

        for p_dir in patient_dirs:
            # 索引目录下所有 .pt 文件
            pt_files = [os.path.join(p_dir, f) for f in os.listdir(p_dir)
                        if f.endswith('.pt')]
            self.samples.extend(pt_files)

        if len(self.samples) == 0:
            print(f"[!] 警告：在 {data_dir} 中未发现有效的 .pt 样本文件。")
        else:
            print(f"[*] 索引构建完成，共锁定 {len(self.samples)} 个 2D 物理样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        filename = os.path.basename(file_path)

        try:
            # ---------------------------------------------------------
            # [核心改造]：利用正则动态解析文件名中的剂量比例 (Dose Ratio)
            # ---------------------------------------------------------
            match = re.search(r'_D(\d+)_', filename)
            if match:
                dose_denominator = float(match.group(1))
                dose_ratio = 1.0 / dose_denominator
            else:
                # 容错降级：如果遇到没有 _D_ 标签的老数据，默认按 1/10 剂量处理
                dose_ratio = 0.1

            # 封装为 (1,) 的一维张量，供主干网络中的 MLP 嵌入层读取
            dose_tensor = torch.tensor([dose_ratio], dtype=torch.float32)

            # 加载 [2, H, W] 物理张量
            tensor_pair = torch.load(file_path, map_location='cpu', weights_only=True)

            # 提取并升维至 (1, H, W) 以适配网络输入通道
            condition_tensor = tensor_pair[0:1, :, :]
            target_tensor = tensor_pair[1:2, :, :]

            # [架构桥接]：返回三元组
            return target_tensor, condition_tensor, dose_tensor

        except Exception as e:
            # 容错处理：若读取失败，尝试加载下一个样本（带递归防护）
            tries = getattr(self, '_getitem_tries', 0)
            if tries > len(self.samples):
                self._getitem_tries = 0
                raise RuntimeError(
                    f"所有 {len(self.samples)} 个样本均加载失败，请检查数据完整性: {self.data_dir}"
                )
            self._getitem_tries = tries + 1
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)
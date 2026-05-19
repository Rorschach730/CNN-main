import os
import torch
import re
from torch.utils.data import Dataset

class PETDatasetTrido(Dataset):
    """
    专为 TriDo-JiT 设计的 PET 数据集读取类。
    返回严格的三元组：(target_tensor, condition_tensor, body_part)
    """
    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据路径不存在: {data_dir}")

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.pt'):
                    self.samples.append(os.path.join(root, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        filename = os.path.basename(file_path)

        try:
            tensor_pair = torch.load(file_path, map_location='cpu', weights_only=True)
            target_tensor = tensor_pair[0].unsqueeze(0).float()
            condition_tensor = tensor_pair[1].unsqueeze(0).float()

            match = re.search(r'_Part(\d+)_', filename)
            if match:
                body_part_idx = int(match.group(1))
            else:
                body_part_idx = 1

            body_part = torch.tensor(body_part_idx, dtype=torch.long)
            return target_tensor, condition_tensor, body_part

        except Exception as e:
            dummy_tensor = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            return dummy_tensor, dummy_tensor, torch.tensor(1, dtype=torch.long)

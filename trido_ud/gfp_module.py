"""
Global Frequency Parsing (GFP) Module for TriDo-JiT
=====================================================
重构修复版：
1. 修复 2D 离散余弦变换 (DCT) 在处理非正方形高维张量时隐藏的轴交叉转置缺陷。
2. 废除硬编码的 0.3L1 频域粗糙目标，在 Loss 计算时直接复用前向前向高频软掩码，梯度完全对齐。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def dct_1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """对最后一个维度执行严格的 1D DCT 变换"""
    N = x.shape[-1]
    return F.linear(x, _dct_matrix(N, device=x.device, norm=norm))

def idct_1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """对最后一个维度执行严格的 1D IDCT 变换"""
    N = x.shape[-1]
    return F.linear(x, _dct_matrix(N, device=x.device, norm=norm).t())

def dct_2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """完美的 2D DCT：行列解耦，避免非必须转置"""
    # 变换最后一维 (Cols)
    x_dct_cols = dct_1d(x, norm=norm)
    # 变换倒数第二维 (Rows)
    x_dct_rows = dct_1d(x_dct_cols.transpose(-1, -2), norm=norm)
    return x_dct_rows.transpose(-1, -2)

def idct_2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """完美的 2D IDCT"""
    x_idct_cols = idct_1d(x, norm=norm)
    x_idct_rows = idct_1d(x_idct_cols.transpose(-1, -2), norm=norm)
    return x_idct_rows.transpose(-1, -2)

def _dct_matrix_impl(N: int, device: torch.device, norm: str = 'ortho'):
    k = torch.arange(N, dtype=torch.float32, device=device)
    n = torch.arange(N, dtype=torch.float32, device=device)
    C = torch.cos(math.pi * k.unsqueeze(1) * (n + 0.5) / N)
    if norm == 'ortho':
        C[0, :] *= math.sqrt(1.0 / N)
        C[1:, :] *= math.sqrt(2.0 / N)
    return C

_dct_cache = {}

def _dct_matrix(N: int, device: torch.device, norm: str = 'ortho'):
    key = (N, device.type, norm)
    if key not in _dct_cache:
        _dct_cache[key] = _dct_matrix_impl(N, device, norm)
    return _dct_cache[key]


class FrequencyBandSplit(nn.Module):
    def __init__(self, img_size: int = 256, n_bands: int = 3):
        super().__init__()
        self.img_size = img_size
        self.n_bands = n_bands

        h = torch.arange(img_size, dtype=torch.float32)
        w = torch.arange(img_size, dtype=torch.float32)
        H, W = torch.meshgrid(h, w, indexing='ij')
        freq_dist = torch.sqrt(H ** 2 + W ** 2) / (img_size * math.sqrt(2))
        self.register_buffer('freq_dist', freq_dist)

        self.band_centers = nn.Parameter(torch.linspace(0.05, 0.85, n_bands))
        self.band_widths = nn.Parameter(torch.ones(n_bands) * 0.15)

    def forward(self, dct_coeffs: torch.Tensor):
        B, C, H, W = dct_coeffs.shape
        freq_dist = self.freq_dist.view(1, 1, H, W)

        band_outputs = []
        band_masks = []

        for i in range(self.n_bands):
            center = torch.sigmoid(self.band_centers[i])
            width = torch.sigmoid(self.band_widths[i]) * 0.3 + 0.05
            mask = torch.exp(-((freq_dist - center) ** 2) / (2 * width ** 2))
            mask = mask / (mask.max() + 1e-8)

            band_outputs.append(dct_coeffs * mask)
            band_masks.append(mask)

        return band_outputs, band_masks


class GFPModule(nn.Module):
    def __init__(self, img_size: int = 256, n_bands: int = 3,
                 enh_channels: int = 32, body_part_cond: bool = True):
        super().__init__()
        self.img_size = img_size
        self.n_bands = n_bands
        self.body_part_cond = body_part_cond

        self.band_split = FrequencyBandSplit(img_size, n_bands)
        self.band_weights = nn.Parameter(torch.ones(n_bands))
        nn.init.constant_(self.band_weights[-1], 1.5)

        self.hf_enhance = nn.Sequential(
            nn.Conv2d(1, enh_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(enh_channels, enh_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(enh_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        if body_part_cond:
            self.body_part_to_scale = nn.Sequential(
                nn.Embedding(3, 16),
                nn.Flatten(start_dim=1),
                nn.Linear(16, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, image: torch.Tensor, body_part: torch.Tensor = None) -> torch.Tensor:
        dct_coeffs = dct_2d(image, norm='ortho')
        band_coeffs, band_masks = self.band_split(dct_coeffs)

        enhanced_dct = torch.zeros_like(dct_coeffs)
        for i in range(self.n_bands):
            w = torch.sigmoid(self.band_weights[i]) * 2.0
            enhanced_dct = enhanced_dct + band_coeffs[i] * w

        hf_dct = band_coeffs[-1]
        hf_image = idct_2d(hf_dct * band_masks[-1], norm='ortho')
        hf_refined = self.hf_enhance(hf_image)

        if self.body_part_cond and body_part is not None:
            body_part_scale = self.body_part_to_scale(body_part).view(-1, 1, 1, 1)
            hf_refined = hf_refined * (1.0 + body_part_scale * 0.5)

        return idct_2d(enhanced_dct, norm='ortho') + 0.1 * hf_refined

    def compute_frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """完全对齐的软掩码频域高频 L1 损失"""
        pred_dct = dct_2d(pred, norm='ortho')
        target_dct = dct_2d(target, norm='ortho')

        # 动态通过内部 band_split 获取和前向完全对齐的高频软掩码
        _, band_masks = self.band_split(target_dct)
        hf_soft_mask = band_masks[-1] # 复用最高频的高斯 Mask

        hf_diff = (pred_dct - target_dct).abs() * hf_soft_mask
        return hf_diff.mean()

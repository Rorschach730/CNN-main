"""
Global Frequency Parsing (GFP) Module v2 for TriDo-JiT
=======================================================
融合版：gfp_module.py 的软掩码对齐 + gfp_module_1.py 的完整文档。
核心改进：compute_frequency_loss 复用 band_split 的软掩码，梯度完全对齐。

架构:
  Image → DCT_2D → FrequencyBandSplit → [Low, Mid, High] bands
  → Per-band learnable weights + HF CNN enhancement
  → IDCT_2D → Enhanced Image

身体部位调节:
  body_part (0=brain, 1=chest, 2=abdomen) → Embedding → MLP → Sigmoid
  → 缩放 HF 增强强度（不同部位对高频细节敏感度不同）

DCT 实现:
  严格行列解耦 2D DCT，避免非正方形张量的隐藏转置缺陷。
  使用缓存 DCT 矩阵，避免重复构建。

Reference: Frequency-domain augmentation in image restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════
# DCT / IDCT 变换
# ═══════════════════════════════════════════════════════════════

def dct_1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """对最后一个维度执行 1D DCT-II 变换"""
    N = x.shape[-1]
    return F.linear(x, _dct_matrix(N, device=x.device, norm=norm))


def idct_1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """对最后一个维度执行 1D IDCT-III 变换（DCT 矩阵的转置）"""
    N = x.shape[-1]
    return F.linear(x, _dct_matrix(N, device=x.device, norm=norm).t())


def dct_2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    2D DCT-II：行列解耦，避免非必须转置。

    先对列 (dim=-2) 做 DCT，再对行 (dim=-1) 做 DCT，
    使用独立的 transpose 操作保证轴正确性。

    Args:
        x: (..., H, W)
        norm: 'ortho' (正交归一) 或 'backward'

    Returns:
        DCT coefficients: (..., H, W)
    """
    # Step 1: DCT along rows (last dim)
    x_dct_rows = dct_1d(x, norm=norm)
    # Step 2: DCT along columns (second-to-last dim)
    x_dct_cols = dct_1d(x_dct_rows.transpose(-1, -2), norm=norm)
    return x_dct_cols.transpose(-1, -2)


def idct_2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    2D IDCT-III：行列解耦逆变换。

    Args:
        x: (..., H, W)
        norm: 'ortho' 或 'backward'

    Returns:
        Spatial domain: (..., H, W)
    """
    x_idct_rows = idct_1d(x, norm=norm)
    x_idct_cols = idct_1d(x_idct_rows.transpose(-1, -2), norm=norm)
    return x_idct_cols.transpose(-1, -2)


# ── DCT 矩阵缓存 ──

def _dct_matrix_impl(N: int, device: torch.device, norm: str = 'ortho') -> torch.Tensor:
    """
    构建 DCT-II 矩阵 C[k,n] = cos(π * k * (n + 0.5) / N)
    正交归一化: C[0] *= √(1/N), C[k>0] *= √(2/N)
    """
    k = torch.arange(N, dtype=torch.float32, device=device)
    n = torch.arange(N, dtype=torch.float32, device=device)
    C = torch.cos(math.pi * k.unsqueeze(1) * (n + 0.5) / N)  # (N, N)
    if norm == 'ortho':
        C[0, :] *= math.sqrt(1.0 / N)
        C[1:, :] *= math.sqrt(2.0 / N)
    return C


_dct_cache: dict = {}


def _dct_matrix(N: int, device: torch.device, norm: str = 'ortho') -> torch.Tensor:
    """缓存的 DCT 矩阵获取器"""
    key = (N, device.type, norm)
    if key not in _dct_cache:
        _dct_cache[key] = _dct_matrix_impl(N, device, norm)
    return _dct_cache[key]


# ═══════════════════════════════════════════════════════════════
# 频带分割
# ═══════════════════════════════════════════════════════════════

class FrequencyBandSplit(nn.Module):
    """
    使用可学习高斯软掩码将 DCT 系数分割为频带。

    与硬阈值不同，软掩码对梯度流更友好，允许网络自适应调节频带边界。

    Args:
        img_size: 输入图像尺寸
        n_bands:  频带数量（默认 3: low, mid, high）
    """

    def __init__(self, img_size: int = 256, n_bands: int = 3):
        super().__init__()
        self.img_size = img_size
        self.n_bands = n_bands

        # 频率距离图: 每个 DCT 系数的归一化径向频率 ∈ [0, 1]
        h = torch.arange(img_size, dtype=torch.float32)
        w = torch.arange(img_size, dtype=torch.float32)
        H, W = torch.meshgrid(h, w, indexing='ij')
        freq_dist = torch.sqrt(H ** 2 + W ** 2) / (img_size * math.sqrt(2))
        self.register_buffer('freq_dist', freq_dist)

        # 可学习频带中心和宽度
        self.band_centers = nn.Parameter(torch.linspace(0.05, 0.85, n_bands))
        self.band_widths = nn.Parameter(torch.ones(n_bands) * 0.15)

    def forward(self, dct_coeffs: torch.Tensor):
        """
        将 DCT 系数按频率分割为多个频带。

        Args:
            dct_coeffs: (B, C, H, W) DCT 系数

        Returns:
            band_outputs: List[(B, C, H, W)] 各频带系数
            band_masks:   List[(1, 1, H, W)] 各频带软掩码
        """
        B, C, H, W = dct_coeffs.shape
        freq_dist = self.freq_dist.view(1, 1, H, W)

        band_outputs = []
        band_masks = []

        for i in range(self.n_bands):
            # Sigmoid 约束到 [0, 1]
            center = torch.sigmoid(self.band_centers[i])
            width = torch.sigmoid(self.band_widths[i]) * 0.3 + 0.05  # [0.05, 0.35]

            # 高斯软掩码
            mask = torch.exp(-((freq_dist - center) ** 2) / (2 * width ** 2))
            mask = mask / (mask.max() + 1e-8)  # 归一化到 max=1

            band_outputs.append(dct_coeffs * mask)
            band_masks.append(mask)

        return band_outputs, band_masks


# ═══════════════════════════════════════════════════════════════
# GFP 主模块
# ═══════════════════════════════════════════════════════════════

class GFPModule(nn.Module):
    """
    Global Frequency Parsing 模块。

    通过 DCT 分解图像到频域，对各频带加权增强，尤其强化高频细节。
    Flow Matching 去噪容易过度平滑，GFP 显式补偿丢失的高频信息。

    工作流:
      1. DCT_2D 变换
      2. FrequencyBandSplit → 3 频带 (Low, Mid, High)
      3. 逐频带可学习权重增强
      4. 高频 CNN 精炼（轻量 3 层 Conv+SiLU+Tanh）
      5. 身体部位自适应缩放
      6. IDCT_2D 逆变换 + 0.1×HF 残差

    Args:
        img_size:      输入图像尺寸
        n_bands:       频带数量
        enh_channels:  HF 增强网络隐藏通道
        body_part_cond: 是否启用身体部位条件调节
    """

    def __init__(self, img_size: int = 256, n_bands: int = 3,
                 enh_channels: int = 32, body_part_cond: bool = True):
        super().__init__()
        self.img_size = img_size
        self.n_bands = n_bands
        self.body_part_cond = body_part_cond

        # ── 频带分割器 ──
        self.band_split = FrequencyBandSplit(img_size, n_bands)

        # ── 逐频带权重（可学习）──
        # 初始: Low=1.0, Mid=1.0, High=1.5 (轻微 HF 增强)
        self.band_weights = nn.Parameter(torch.ones(n_bands))
        nn.init.constant_(self.band_weights[-1], 1.5)

        # ── 高频精炼网络 ──
        # 轻量 CNN: 3×3 卷积 + SiLU 激活
        self.hf_enhance = nn.Sequential(
            nn.Conv2d(1, enh_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(enh_channels, enh_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(enh_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # 输出 ∈ [-1, 1]，作为残差
        )

        # ── 身体部位条件调节 ──
        if body_part_cond:
            self.body_part_to_scale = nn.Sequential(
                nn.Embedding(3, 16),       # 3 类: brain/chest/abdomen
                nn.Flatten(start_dim=1),
                nn.Linear(16, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),              # 输出 ∈ [0, 1]
            )

    def forward(self, image: torch.Tensor,
                body_part: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播：频率增强。

        Args:
            image:     (B, 1, H, W) 输入图像
            body_part: (B,) LongTensor 身体部位 (0=brain, 1=chest, 2=abdomen)

        Returns:
            enhanced: (B, 1, H, W) 频率增强图像
        """
        B, C, H, W = image.shape

        # Step 1: DCT 分解
        dct_coeffs = dct_2d(image, norm='ortho')  # (B, C, H, W)

        # Step 2: 频带分割
        band_coeffs, band_masks = self.band_split(dct_coeffs)

        # Step 3: 逐频带加权
        enhanced_dct = torch.zeros_like(dct_coeffs)
        for i in range(self.n_bands):
            w = torch.sigmoid(self.band_weights[i]) * 2.0  # 缩放到 [0, 2]
            enhanced_dct = enhanced_dct + band_coeffs[i] * w

        # Step 4: 高频精炼（从最高频带提取）
        hf_dct = band_coeffs[-1]                         # 最高频带
        hf_image = idct_2d(hf_dct * band_masks[-1], norm='ortho')
        hf_refined = self.hf_enhance(hf_image)           # 可学习 HF 调整

        # Step 5: 身体部位自适应缩放
        if self.body_part_cond and body_part is not None:
            body_part_scale = self.body_part_to_scale(body_part)    # (B, 1)
            body_part_scale = body_part_scale.view(-1, 1, 1, 1)
            hf_refined = hf_refined * (1.0 + body_part_scale * 0.5)

        # Step 6: IDCT 重建 + HF 残差
        enhanced_recon = idct_2d(enhanced_dct, norm='ortho')
        output = enhanced_recon + 0.1 * hf_refined  # 保守 HF 增强

        return output

    def compute_frequency_loss(self, pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
        """
        频域损失：高频软掩码 L1。

        [关键改进] 复用 band_split 获取的高斯软掩码，
        确保与 forward 中的频带定义完全对齐，梯度一致。

        Args:
            pred:   (B, 1, H, W) 预测图像
            target: (B, 1, H, W) 目标图像

        Returns:
            freq_loss: 标量
        """
        # DCT 变换
        pred_dct = dct_2d(pred, norm='ortho')
        target_dct = dct_2d(target, norm='ortho')

        # 通过 band_split 获取与 forward 对齐的高频软掩码
        _, band_masks = self.band_split(target_dct)
        hf_soft_mask = band_masks[-1]  # 最高频带的高斯软掩码

        # 加权 L1: 仅惩罚高频区域的差异
        hf_diff = (pred_dct - target_dct).abs() * hf_soft_mask
        return hf_diff.mean()

"""
TriDo-Simple: 三域简化解耦 CNN 去噪（无扩散模型）
==================================================
纯粹的前馈三域级联，去掉 Flow Matching 扩散范式：
  - 无时间步采样
  - 无 ODE 求解器
  - 无 CFG 引导
  - 直接 condition → output 映射

三域架构:
  ① Sinogram Domain  — Radon → ResNetSino → FBP
  ② Image Domain      — ResNetDenoiser (直接去噪)
  ③ Frequency Domain  — ResNetFreq (频域增强)

与 TriDo-CNN (扩散版) 的核心区别:
  - Image Domain: Flow Matching ODE → 直接 ResNet 前馈
  - Frequency Domain: GFP DCT → 简单 ResNet
  - 不需要 timestep embedding、body_part embedding
  - 训练: 直接 L1 loss，无辅助损失
  - 推理: model(condition) 一次前向即得结果

命名规范: 所有文件以 _simple 结尾，与 trido_ud/ 区分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# 复用 sino bridge（网格已缓存版）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from trido_ud.radon_transform import SinoImageBridge


# ==============================================================================
# 通用 ResBlock（简洁版，无 FiLM 条件注入）
# ==============================================================================

class SimpleResBlock(nn.Module):
    """纯残差块: GroupNorm → SiLU → Conv3x3 → GroupNorm → SiLU → Conv3x3 + skip"""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        g = min(groups, channels) if channels >= groups else channels
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


# ==============================================================================
# Domain ①: Sinogram ResNet
# ==============================================================================

class ResNetSino(nn.Module):
    """
    Sinogram 域简单 ResNet 去噪。
    输入: (B, 1, n_views, det_size) 原始 sinogram
    输出: (B, 1, n_views, det_size) 去噪 sinogram
    """

    def __init__(self, n_views: int = 96, det_size: int = 256,
                 base_ch: int = 32, n_blocks: int = 6):
        super().__init__()
        self.conv_in = nn.Conv2d(1, base_ch, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[
            SimpleResBlock(base_ch) for _ in range(n_blocks)
        ])
        self.norm_out = nn.GroupNorm(min(8, base_ch), base_ch)
        self.conv_out = nn.Conv2d(base_ch, 1, 3, padding=1)

        # 零初始化输出层（残差学习：初始预测 delta=0）
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, sino):
        x = self.conv_in(sino)
        x = self.blocks(x)
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return sino + x  # 残差连接


# ==============================================================================
# Domain ②: Image ResNet Denoiser
# ==============================================================================

class ResNetDenoiser(nn.Module):
    """
    图像域直接去噪 ResNet（替代 Flow Matching 扩散）。
    输入: (B, 2, H, W) — concat(condition, image_from_sino) 或 concat(condition, condition)
    输出: (B, 1, H, W) — 干净图像
    
    使用 gradient checkpoint 控制显存 (12 blocks × 256² 全分辨率)。
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1,
                 base_ch: int = 64, n_blocks: int = 12):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False)
        self.blocks = nn.ModuleList([
            SimpleResBlock(base_ch) for _ in range(n_blocks)
        ])
        self.norm_out = nn.GroupNorm(min(8, base_ch), base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_channels, 3, padding=1)

        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x


# ==============================================================================
# Domain ③: Frequency ResNet
# ==============================================================================

class ResNetFreq(nn.Module):
    """
    频域增强 ResNet（替代 GFP DCT）。
    输入: (B, 1, H, W) 去噪图像
    输出: (B, 1, H, W) 频域增强图像
    """

    def __init__(self, base_ch: int = 32, n_blocks: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(1, base_ch, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[
            SimpleResBlock(base_ch) for _ in range(n_blocks)
        ])
        self.norm_out = nn.GroupNorm(min(8, base_ch), base_ch)
        self.conv_out = nn.Conv2d(base_ch, 1, 3, padding=1)

        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x


# ==============================================================================
# TriDo-Simple: 三域简化解耦 CNN
# ==============================================================================

class TriDoSimpleCNN(nn.Module):
    """
    三域简化解耦 CNN：Sinogram → Image → Frequency 级联去噪。

    架构:
      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
      │ Radon → Sino │→  │ FBP → Image  │→  │  Freq        │
      │   ResNet     │   │   ResNet      │   │  ResNet      │
      └──────────────┘   └──────────────┘   └──────────────┘

    Args:
        input_size: 输入图像尺寸
        n_views:    Radon 投影视角数
        det_size:   探测器 bin 数
        sino_ch:    Sinogram ResNet 基础通道
        sino_blocks: Sinogram ResNet 残差块数
        img_ch:     Image ResNet 基础通道
        img_blocks: Image ResNet 残差块数
        freq_ch:    Frequency ResNet 基础通道
        freq_blocks: Frequency ResNet 残差块数
        use_sino_domain:  启用 sinogram 域
        use_freq_domain:  启用 frequency 域
    """

    def __init__(self, input_size: int = 256,
                 n_views: int = 96, det_size: int = 256,
                 sino_ch: int = 32, sino_blocks: int = 6,
                 img_ch: int = 64, img_blocks: int = 12,
                 freq_ch: int = 32, freq_blocks: int = 4,
                 use_sino_domain: bool = True,
                 use_freq_domain: bool = True):
        super().__init__()
        self.input_size = input_size
        self.n_views = n_views
        self.use_sino_domain = use_sino_domain
        self.use_freq_domain = use_freq_domain

        # --- Domain ①: Sinogram ---
        if use_sino_domain:
            self.sino_bridge = SinoImageBridge(
                n_views=n_views, img_size=input_size, det_size=det_size
            )
            self.sino_resnet = ResNetSino(
                n_views=n_views, det_size=det_size,
                base_ch=sino_ch, n_blocks=sino_blocks
            )
        else:
            self.sino_bridge = None
            self.sino_resnet = None

        # --- Domain ②: Image ---
        in_ch = 2 if use_sino_domain else 1
        self.image_resnet = ResNetDenoiser(
            in_channels=in_ch, out_channels=1,
            base_ch=img_ch, n_blocks=img_blocks
        )

        # --- Domain ③: Frequency ---
        if use_freq_domain:
            self.freq_resnet = ResNetFreq(
                base_ch=freq_ch, n_blocks=freq_blocks
            )
        else:
            self.freq_resnet = None

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        前馈去噪（无时间步、无 ODE）。

        Args:
            condition: (B, 1, H, W) 低剂量 PET 图像

        Returns:
            output: (B, 1, H, W) 去噪图像
        """
        # ================================================================
        # Domain ①: Sinogram
        # ================================================================
        if self.use_sino_domain:
            sino = self.sino_bridge.forward_project(condition)
            sino = self.sino_resnet(sino)
            image_from_sino = self.sino_bridge.reconstruct(sino)
            image_input = torch.cat([condition, image_from_sino], dim=1)
        else:
            image_input = condition

        # ================================================================
        # Domain ②: Image (直接映射，无需扩散)
        # ================================================================
        output = self.image_resnet(image_input)

        # ================================================================
        # Domain ③: Frequency
        # ================================================================
        if self.use_freq_domain:
            output = output + self.freq_resnet(output)  # 残差增强

        return output


# ==============================================================================
# 模型工厂
# ==============================================================================

def TriDoSimple_Base(**kwargs):
    """Base: img_ch=64, img_blocks=12, ~4.5M params"""
    return TriDoSimpleCNN(**kwargs)


def TriDoSimple_Small(**kwargs):
    """Small: img_ch=32, img_blocks=8, ~1.5M params"""
    return TriDoSimpleCNN(img_ch=32, img_blocks=8, sino_ch=24, freq_ch=24, **kwargs)


def TriDoSimple_Tiny(**kwargs):
    """Tiny: img_ch=24, img_blocks=6, ~0.8M params"""
    return TriDoSimpleCNN(img_ch=24, img_blocks=6, sino_ch=16, freq_ch=16,
                          sino_blocks=4, freq_blocks=2, **kwargs)


TriDoSimple_models = {
    'TriDoSimple-Base': TriDoSimple_Base,
    'TriDoSimple-Small': TriDoSimple_Small,
    'TriDoSimple-Tiny': TriDoSimple_Tiny,
}

"""
TriDo-CNN: Triple-Domain CNN for PET Denoising
================================================
Three-domain architecture:
  1. Sinogram Domain — sino encoder with body-part conditioning (Cross-Domain Recon)
  2. Image Domain     — ResNet U-Net CNN refinement (Flow Matching v-prediction)
  3. Frequency Domain — GFP high-frequency enhancement (DCT-based)

The three domains operate in a cascade:
  Input image → Radon → Sinogram → SinoEncoder → FBP → ResNet CNN → GFP → Output

Key change from JiT-main: Domain ② backbone upgraded from JiT Transformer
(Attention-based) to ResNet U-Net CNN (Convolution-based), providing:
  - Inductive bias better suited for medical image denoising
  - Lower memory footprint during inference
  - No quadratic complexity in spatial dimensions
  - Proven effectiveness on PET/CT reconstruction tasks

References:
  - ResNet: https://arxiv.org/abs/1512.03385
  - U-Net: https://arxiv.org/abs/1505.04597
  - Cross-Domain Reconstruction (sinogram domain)
  - Prior Knowledge-Guided Triple-Domain Transformer-GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# ==============================================================================
# Timestep Embedding (kept for Flow Matching conditioning)
# ==============================================================================

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding for Flow Matching."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * 1000.0, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ==============================================================================
# Domain ②: ResNet U-Net CNN Backbone
# ==============================================================================

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation for conditioning injection.
    γ(cond) * norm(x) + β(cond)
    Zero-initialized: starts as identity, learns modulation gradually.
    """

    def __init__(self, n_channels: int, cond_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, n_channels)
        self.beta_proj = nn.Linear(cond_dim, n_channels)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(cond).unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU + conditioning via FiLM.
    Two 3×3 convolutions with conditioning injected before each one.
    """

    def __init__(self, channels: int, cond_dim: int, dropout: float = 0.0,
                 groups: int = 8):
        super().__init__()
        # Adjust groups if channels < groups
        actual_groups = min(groups, channels) if channels >= groups else channels

        self.norm1 = nn.GroupNorm(actual_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.film1 = FiLM(channels, cond_dim)

        self.norm2 = nn.GroupNorm(actual_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.film2 = FiLM(channels, cond_dim)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.film1(h, cond)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.film2(h, cond)
        h = F.silu(h)
        h = self.conv2(h)

        h = self.dropout(h)
        return x + h


class DownBlock(nn.Module):
    """Resolution halving: 升维 → ResBlocks → 下采样 (stride=2)"""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 num_res_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        # 先升维到 out_ch（替代原来第一个 ResBlock 使用 in_ch）
        self.conv_in = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        # 所有 ResBlock 均使用 out_ch 通道
        self.res_blocks = nn.ModuleList([
            ResBlock(out_ch, cond_dim, dropout) for _ in range(num_res_blocks)
        ])
        # 下采样（stride=2，保持通道数不变）
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x, cond):
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = checkpoint(block, x, cond, use_reentrant=False)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Resolution doubling: bilinear upsample + concat skip → ResBlocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int,
                 num_res_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # After concat: in_ch + skip_ch → out_ch
        self.conv_in = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.res_blocks = nn.ModuleList([
            ResBlock(out_ch, cond_dim, dropout) for _ in range(num_res_blocks)
        ])

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = checkpoint(block, x, cond, use_reentrant=False)
        return x


class ResNetUNet(nn.Module):
    """
    ResNet U-Net CNN backbone for PET denoising (Domain ②).

    Architecture:
      Encoder:  4 stages (64 → 128 → 256 → 512 channels)
      Bottleneck: 2 ResBlocks at 512 channels
      Decoder:   4 stages (512 → 256 → 128 → 64 channels)
      Skip connections from each encoder stage to corresponding decoder stage.
      FiLM conditioning at every ResBlock (timestep + body_part).

    Args:
        in_channels: Input channels (z + condition = 2)
        out_channels: Output channels (denoised image = 1)
        base_ch: Base channel count (controls model capacity)
        channel_mult: Channel multipliers per stage
        num_res_blocks: ResBlocks per encoder/decoder stage
        cond_dim: Conditioning embedding dimension
        dropout: Dropout rate
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1,
                 base_ch: int = 64, channel_mult: tuple = (1, 2, 4, 8),
                 num_res_blocks: int = 2, cond_dim: int = 768,
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_ch = base_ch
        self.cond_dim = cond_dim

        # --- Initial convolution ---
        self.conv_in = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False)

        # --- Encoder ---
        chs = [base_ch]
        in_ch = base_ch
        self.down_blocks = nn.ModuleList([])
        for mult in channel_mult:
            out_ch = base_ch * mult
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, cond_dim, num_res_blocks, dropout)
            )
            chs.append(out_ch)
            in_ch = out_ch

        # --- Bottleneck ---
        bottleneck_ch = base_ch * channel_mult[-1]
        self.bottleneck = nn.ModuleList([
            ResBlock(bottleneck_ch, cond_dim, dropout)
            for _ in range(num_res_blocks)
        ])

        # --- Decoder (修正部分) ---
        # 所有 encoder skip 的通道（从第一个 down block 到最后一个）
        encoder_skip_chs = chs[1:]   # 例如 [128, 256, 512, 1024] (取决于 base_ch)
        decoder_chs = list(reversed(encoder_skip_chs))  # 从深到浅

        self.up_blocks = nn.ModuleList([])
        for i in range(len(channel_mult)):
            skip_ch = decoder_chs[i]
            # 输出通道：逐渐降回 base_ch
            if i < len(channel_mult) - 1:
                out_ch = base_ch * channel_mult[-(i + 2)]
            else:
                out_ch = base_ch
            in_ch_up = in_ch  # 来自上一层的通道
            if i == 0:
                in_ch_up = bottleneck_ch
            self.up_blocks.append(
                UpBlock(in_ch_up, skip_ch, out_ch, cond_dim, num_res_blocks, dropout)
            )
            in_ch = out_ch

        # --- Output convolution ---
        self.norm_out = nn.GroupNorm(min(8, base_ch), base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, cond):
        x = self.conv_in(x)
        skips = []
        for down in self.down_blocks:
            x, skip = down(x, cond)
            skips.append(skip)
        for block in self.bottleneck:
            x = checkpoint(block, x, cond, use_reentrant=False)
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            x = up(x, skip, cond)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# ==============================================================================
# TriDo-CNN: Three-Domain Architecture (CNN version)
# ==============================================================================

# TriDo-specific modules (support both package and direct invocation)
try:
    from trido_ud.sino_processor import SinogramEncoder
    from trido_ud.radon_transform import SinoImageBridge
    from trido_ud.gfp_module import GFPModule
except ImportError:
    from sino_processor import SinogramEncoder
    from radon_transform import SinoImageBridge
    from gfp_module import GFPModule


class TriDoCNN(nn.Module):
    """
    Triple-Domain CNN for PET Denoising.

    Three domains cascaded:
      ┌─────────────────────────────────────────────────────────────┐
      │  ① Sinogram Domain   ② Image Domain     ③ Frequency Domain  │
      │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
      │  │ Radon → Sino │→  │ FBP → ResNet │→  │ GFP Enhance  │→   │
      │  │   Encoder    │   │   U-Net CNN  │   │   Module     │    │
      │  └──────────────┘   └──────────────┘   └──────────────┘    │
      └─────────────────────────────────────────────────────────────┘

    Key features:
      - Body part embedding injected at sinogram AND image domain
      - FBP provides differentiable sinogram→image bridge
      - ResNet U-Net CNN backbone with FiLM conditioning
      - GFP enhances high-frequencies lost in denoising

    Args:
        input_size: Input image size (square)
        patch_size: Kept for API compatibility (not used by CNN backbone)
        in_channels: Input channels (condition + noisy = 2)
        out_channels: Output channels (denoised image = 1)
        hidden_size: Conditioning embedding dimension (controls FiLM capacity)
        base_ch: CNN base channel count
        num_res_blocks: ResBlocks per encoder/decoder stage
        attn_drop: Kept for API compatibility (not used by CNN)
        proj_drop: Dropout rate for CNN ResBlocks
        n_views: Sinogram projection views
        sino_base_ch: Sinogram encoder base channels
        use_sino_domain: Enable sinogram domain processing
        use_freq_domain: Enable frequency domain enhancement
    """

    def __init__(self, input_size=256, patch_size=16, in_channels=2, out_channels=1,
                 hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, bottleneck_dim=128,
                 base_ch=64, num_res_blocks=2,
                 n_views=128, sino_base_ch=32,
                 use_sino_domain=True, use_freq_domain=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_sino_domain = use_sino_domain
        self.use_freq_domain = use_freq_domain

        # --- Time embedding ---
        self.t_embedder = TimestepEmbedder(hidden_size)

        # --- Body part embedding (3 classes: 0=brain, 1=chest, 2=abdomen) ---
        self.body_part_embedder = nn.Embedding(3, hidden_size)

        # --- Domain ①: Sinogram ---
        if use_sino_domain:
            self.sino_bridge = SinoImageBridge(
                n_views=n_views, img_size=input_size,
                det_size=input_size
            )
            self.sino_encoder = SinogramEncoder(
                n_views=n_views, n_bins=input_size,
                base_channels=sino_base_ch, cond_dim=128
            )
        else:
            self.sino_bridge = None
            self.sino_encoder = None

        # --- Domain ②: Image (ResNet U-Net CNN) ---
        # CNN operates directly on 2D images, no patchification needed.
        self.cnn = ResNetUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_ch=base_ch,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=num_res_blocks,
            cond_dim=hidden_size,
            dropout=proj_drop,
        )

        # --- Domain ③: Frequency (GFP) ---
        if use_freq_domain:
            self.gfp = GFPModule(img_size=input_size, body_part_cond=True)
        else:
            self.gfp = None

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with standard practices for CNNs."""

        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Timestep embedder special init
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Body part embedder normal init
        nn.init.normal_(self.body_part_embedder.weight, std=0.02)

        # Output layer: zero init for residual learning (predict zero delta at start)
        nn.init.constant_(self.cnn.conv_out.weight, 0)
        nn.init.constant_(self.cnn.conv_out.bias, 0)

    def forward(self, model_input, t, body_part):
        """
        Full three-domain forward pass.

        Args:
            model_input: (B, 2, H, W) — concat(z, condition)
            t: (B,) — flow matching timestep
            body_part: (B,) LongTensor — body part category (0=brain, 1=chest, 2=abdomen)

        Returns:
            output: (B, 1, H, W) — predicted clean image
        """
        B, C, H, W = model_input.shape

        # --- Body part conditioning ---
        body_part_emb = self.body_part_embedder(body_part)   # (B, hidden_size)

        # --- Timestep conditioning ---
        t_emb = self.t_embedder(t)

        # [核心融合]：时间上下文与身体部位上下文物理叠加
        c = t_emb + body_part_emb

        # --- Extract z (noisy) and condition separately ---
        z = model_input[:, 0:1, :, :]       # (B, 1, H, W) — noisy image
        cond = model_input[:, 1:2, :, :]    # (B, 1, H, W) — condition image

        # ================================================================
        # Domain ①: Sinogram Domain Processing
        # ================================================================
        if self.use_sino_domain:
            # Forward project condition to sinogram domain
            sino_cond = self.sino_bridge.forward_project(cond)  # (B, 1, n_views, det_size)

            # Sino encoder denoising with body part conditioning
            sino_denoised = self.sino_encoder(sino_cond, body_part)  # (B, 1, n_views, det_size)

            # FBP reconstruction back to image domain
            image_from_sino = self.sino_bridge.reconstruct(sino_denoised)  # (B, 1, H, W)

            # Combine FBP output with original z for the image domain
            image_input = torch.cat([z, image_from_sino], dim=1)  # (B, 2, H, W)
        else:
            # Without sino domain, use z + cond directly
            image_input = torch.cat([z, cond], dim=1)

        # ================================================================
        # Domain ②: Image Domain (ResNet U-Net CNN)
        # ================================================================
        image_output = self.cnn(image_input, c)  # (B, 1, H, W)

        # ================================================================
        # Domain ③: Frequency Domain (GFP Enhancement)
        # ================================================================
        if self.use_freq_domain:
            output = self.gfp(image_output, body_part)
        else:
            output = image_output

        return output


# ==============================================================================
# Model Factory Functions
# ==============================================================================

def TriDoCNN_Large(**kwargs):
    """TriDo-CNN Large: base_ch=128, 3 res_blocks/stage (~52M params)."""
    patch_size = kwargs.pop('patch_size', 16)
    return TriDoCNN(
        hidden_size=1024, base_ch=128, num_res_blocks=3,
        patch_size=patch_size, **kwargs
    )


def TriDoCNN_Base(**kwargs):
    """TriDo-CNN Base: base_ch=64, 2 res_blocks/stage (~14M params)."""
    patch_size = kwargs.pop('patch_size', 16)
    return TriDoCNN(
        hidden_size=768, base_ch=64, num_res_blocks=2,
        patch_size=patch_size, **kwargs
    )


def TriDoCNN_Small(**kwargs):
    """TriDo-CNN Small: base_ch=32, 1 res_block/stage (~3M params)."""
    patch_size = kwargs.pop('patch_size', 16)
    return TriDoCNN(
        hidden_size=512, base_ch=32, num_res_blocks=1,
        patch_size=patch_size, **kwargs
    )


# Factory dictionary — backward compatible key naming
TriDoCNN_models = {
    'TriDoCNN-Large': TriDoCNN_Large,
    'TriDoCNN-Base': TriDoCNN_Base,
    'TriDoCNN-Small': TriDoCNN_Small,
}

# Legacy alias for backward compatibility with checkpoints that use the old key format
TriDoJiT_models = TriDoCNN_models

"""
Sinogram Domain Processor for TriDo-JiT
========================================
Implements sinogram-domain denoising with body-part conditioning.
Based on the Cross-Domain Reconstruction approach: processes raw sinogram
data in the projection domain before FBP reconstruction.

Architecture:
  - Conv2D-based encoder-decoder (U-Net style) operating on sinograms
  - Body part conditioning via FiLM (Feature-wise Linear Modulation)
  - Residual connections for stable gradient flow
  - Lightweight design: most heavy lifting done by JiT in image domain

Reference: Cross-Domain Reconstruction.pdf — Sinogram domain processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation for body part conditioning.
    γ(body_part) * x + β(body_part)
    """

    def __init__(self, n_channels: int, condition_dim: int = 128):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, n_channels)
        self.beta_proj = nn.Linear(condition_dim, n_channels)

        # Initialize: gamma ≈ 1, beta ≈ 0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
            condition: (B, cond_dim) conditioning vector
        """
        gamma = self.gamma_proj(condition).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_proj(condition).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return (1.0 + gamma) * x + beta


class SinoConvBlock(nn.Module):
    """Convolution block for sinogram processing: Conv → FiLM → Norm → Act"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.film = FiLM(out_ch, cond_dim)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.film(x, cond)
        x = self.norm(x)
        x = self.act(x)
        return x


class SinogramEncoder(nn.Module):
    """
    Sinogram-domain denoising encoder.

    Takes a sinogram of shape (B, 1, n_views, n_bins) and produces a
    denoised sinogram of the same shape. Uses body-part conditioning
    to adapt the denoising strategy to the anatomical region.

    Architecture: Lightweight U-Net style with FiLM conditioning.
    The sinogram has a different spatial structure than images:
      - Vertical axis: projection angles (views)
      - Horizontal axis: detector bins
    Convolutions capture local correlations in both dimensions.

    Args:
        n_views: Number of projection angles
        n_bins: Number of detector bins
        base_channels: Base channel count
        cond_dim: Body part condition embedding dimension
        depth: Number of down/up blocks
    """

    def __init__(self, n_views: int = 256, n_bins: int = 256,
                 base_channels: int = 32, cond_dim: int = 128, depth: int = 3):
        super().__init__()
        self.n_views = n_views
        self.n_bins = n_bins
        self.depth = depth

        # Channel progression
        ch = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Body part conditioning: embed category (0/1/2) to condition vector
        self.body_part_proj = nn.Sequential(
            nn.Embedding(3, cond_dim // 4),
            nn.Flatten(start_dim=1),
            nn.Linear(cond_dim // 4, cond_dim, bias=False),
            nn.SiLU(),
        )

        # --- Encoder (downsampling) ---
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.enc_blocks.append(SinoConvBlock(1, ch[0], cond_dim))
        for i in range(depth):
            self.downs.append(nn.Conv2d(ch[i], ch[i], kernel_size=2, stride=2, bias=False))
            self.enc_blocks.append(SinoConvBlock(ch[i], ch[i + 1], cond_dim))

        # --- Bottleneck ---
        self.bottleneck = SinoConvBlock(ch[-1], ch[-1], cond_dim)

        # --- Decoder (upsampling) ---
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(depth, 0, -1):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(ch[i], ch[i - 1], kernel_size=1, bias=False)
                )
            )
            self.dec_blocks.append(SinoConvBlock(ch[i - 1] * 2, ch[i - 1], cond_dim))

        # --- Output ---
        self.output_conv = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch[0]), ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], 1, kernel_size=1)
        )

    def forward(self, sinogram: torch.Tensor, body_part: torch.Tensor) -> torch.Tensor:
        """
        Denoise in sinogram domain.

        Args:
            sinogram: (B, 1, n_views, n_bins) — noisy sinogram
            body_part: (B,) LongTensor — body part category (0=brain, 1=chest, 2=abdomen)

        Returns:
            denoised_sinogram: (B, 1, n_views, n_bins)
        """
        # Compute body part conditioning vector
        cond = self.body_part_proj(body_part)  # (B, cond_dim)

        # --- Encoder ---
        skips = []
        x = sinogram
        for i, block in enumerate(self.enc_blocks):
            x = block(x, cond)
            skips.append(x)
            if i < self.depth:
                x = self.downs[i](x)

        # --- Bottleneck ---
        x = self.bottleneck(x, cond)

        # --- Decoder ---
        for i, (up, block) in enumerate(zip(self.ups, self.dec_blocks)):
            x = up(x)
            skip = skips[-(i + 2)]  # Match corresponding encoder skip
            # Pad/crop to match sizes
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = block(x, cond)

        # --- Output ---
        residual = self.output_conv(x)

        # Residual connection: denoised = noisy + correction
        sino_denoised = sinogram + residual

        return sino_denoised

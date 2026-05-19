"""
Global Frequency Parsing (GFP) Module for TriDo-JiT
=====================================================
Implements frequency-domain enhancement using DCT (Discrete Cosine Transform).
Decomposes images into low, mid, and high frequency bands, applies learnable
enhancement to high frequencies, and reconstructs the enhanced image.

This addresses the well-known issue that diffusion/flow-based models tend to
over-smooth fine details (high frequencies). The GFP module explicitly
enhances high-frequency components that are critical for PET image quality
(lesion boundaries, small structures).

Reference: The GFP concept is inspired by frequency-domain augmentation
techniques in image restoration literature, adapted for PET denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def dct_2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    2D Discrete Cosine Transform (Type-II).

    Uses the separable property: DCT_2D = DCT_1D_rows(DCT_1D_cols(x))

    Args:
        x: (B, C, H, W)
        norm: 'ortho' for orthonormal, 'backward' for standard

    Returns:
        DCT coefficients: (B, C, H, W)
    """
    # DCT along rows (last dim)
    x = F.linear(
        x.transpose(-1, -2) if x.dim() == 4 else x,
        _dct_matrix(x.shape[-1], device=x.device, norm=norm)
    )
    if x.dim() == 4:
        x = x.transpose(-1, -2)

    # DCT along cols (second-to-last dim)
    x = F.linear(
        x.transpose(-2, -1) if x.dim() == 4 else x,
        _dct_matrix(x.shape[-2], device=x.device, norm=norm)
    )
    if x.dim() == 4:
        x = x.transpose(-2, -1)

    return x


def idct_2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    2D Inverse DCT (Type-III).
    Uses the transpose of the DCT matrix (IDCT = DCT^T for orthonormal).
    """
    # IDCT along cols
    x = F.linear(
        x.transpose(-2, -1) if x.dim() == 4 else x,
        _dct_matrix(x.shape[-2], device=x.device, norm=norm).t()
    )
    if x.dim() == 4:
        x = x.transpose(-2, -1)

    # IDCT along rows
    x = F.linear(
        x.transpose(-1, -2) if x.dim() == 4 else x,
        _dct_matrix(x.shape[-1], device=x.device, norm=norm).t()
    )
    if x.dim() == 4:
        x = x.transpose(-1, -2)

    return x


def _dct_matrix_impl(N: int, device: torch.device, norm: str = 'ortho'):
    """Build DCT Type-II matrix."""
    k = torch.arange(N, dtype=torch.float32, device=device)
    n = torch.arange(N, dtype=torch.float32, device=device)
    # DCT-II: C[k, n] = cos(π * k * (n + 0.5) / N)
    C = torch.cos(math.pi * k.unsqueeze(1) * (n + 0.5) / N)
    if norm == 'ortho':
        C[0, :] *= math.sqrt(1.0 / N)
        C[1:, :] *= math.sqrt(2.0 / N)
    return C


_dct_cache = {}


def _dct_matrix(N: int, device: torch.device, norm: str = 'ortho'):
    """Cached DCT matrix builder."""
    key = (N, device.type, norm)
    if key not in _dct_cache:
        _dct_cache[key] = _dct_matrix_impl(N, device, norm)
    return _dct_cache[key]


class FrequencyBandSplit(nn.Module):
    """
    Split DCT coefficients into low, mid, and high frequency bands.

    Using a learnable soft split with Gaussian masks rather than hard cutoffs,
    which is more stable for gradient flow.
    """

    def __init__(self, img_size: int = 256, n_bands: int = 3):
        super().__init__()
        self.img_size = img_size
        self.n_bands = n_bands

        # Frequency distance map: distance from DC (0,0) in DCT space
        h = torch.arange(img_size, dtype=torch.float32)
        w = torch.arange(img_size, dtype=torch.float32)
        H, W = torch.meshgrid(h, w, indexing='ij')
        # Normalized frequency distance in [0, 1]
        freq_dist = torch.sqrt(H ** 2 + W ** 2) / (img_size * math.sqrt(2))
        self.register_buffer('freq_dist', freq_dist)

        # Learnable band centers and widths
        self.band_centers = nn.Parameter(torch.linspace(0.05, 0.85, n_bands))
        self.band_widths = nn.Parameter(torch.ones(n_bands) * 0.15)

    def forward(self, dct_coeffs: torch.Tensor):
        """
        Split DCT coefficients into frequency bands using soft masks.

        Args:
            dct_coeffs: (B, C, H, W) — DCT coefficients

        Returns:
            band_coeffs: List of (B, C, H, W) tensors, one per band
            band_masks: List of (1, 1, H, W) soft masks
        """
        B, C, H, W = dct_coeffs.shape
        freq_dist = self.freq_dist.view(1, 1, H, W)  # (1, 1, H, W)

        band_outputs = []
        band_masks = []

        for i in range(self.n_bands):
            center = torch.sigmoid(self.band_centers[i])
            width = torch.sigmoid(self.band_widths[i]) * 0.3 + 0.05  # [0.05, 0.35]

            # Gaussian soft mask
            mask = torch.exp(-((freq_dist - center) ** 2) / (2 * width ** 2))
            # Normalize to max=1
            mask = mask / (mask.max() + 1e-8)

            band_outputs.append(dct_coeffs * mask)
            band_masks.append(mask)

        return band_outputs, band_masks


class GFPModule(nn.Module):
    """
    Global Frequency Parsing module.

    Decomposes the image into frequency bands via DCT, applies learnable
    enhancement to high-frequency components, and reconstructs.

    The key insight: diffusion/flow models tend to blur fine details.
    GFP explicitly boosts high frequencies while preserving low-frequency
    structure (which the image-domain model handles well).

    Args:
        img_size: Input image size
        n_bands: Number of frequency bands (default: 3 = low, mid, high)
        enh_channels: Hidden channels for the enhancement network
        body_part_cond: Whether to use body-part conditioning
    """

    def __init__(self, img_size: int = 256, n_bands: int = 3,
                 enh_channels: int = 32, body_part_cond: bool = True):
        super().__init__()
        self.img_size = img_size
        self.n_bands = n_bands
        self.body_part_cond = body_part_cond

        # Frequency band splitter
        self.band_split = FrequencyBandSplit(img_size, n_bands)

        # Learnable per-band enhancement weights
        # Shape: (n_bands,) — scalar multiplier per band
        self.band_weights = nn.Parameter(torch.ones(n_bands))
        # Initialize: low=1.0, mid=1.0, high=1.5 (slight HF boost)
        nn.init.constant_(self.band_weights[-1], 1.5)

        # Lightweight enhancement network for high-frequency refinement
        self.hf_enhance = nn.Sequential(
            nn.Conv2d(1, enh_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(enh_channels, enh_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(enh_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # Bounded enhancement in [-1, 1]
        )

        # Body part conditioning for enhancement strength
        if body_part_cond:
            self.body_part_to_scale = nn.Sequential(
                nn.Embedding(3, 16),
                nn.Flatten(start_dim=1),
                nn.Linear(16, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Scale in [0, 1]
            )

    def forward(self, image: torch.Tensor, body_part: torch.Tensor = None) -> torch.Tensor:
        """
        Enhance high-frequency components of the image.

        Args:
            image: (B, 1, H, W) — input image
            body_part: (B,) LongTensor — body part category (0=brain, 1=chest, 2=abdomen)

        Returns:
            enhanced: (B, 1, H, W) — frequency-enhanced image
        """
        B, C, H, W = image.shape

        # Step 1: DCT decomposition
        dct_coeffs = dct_2d(image, norm='ortho')  # (B, C, H, W)

        # Step 2: Split into bands
        band_coeffs, band_masks = self.band_split(dct_coeffs)

        # Step 3: Apply per-band weights
        enhanced_dct = torch.zeros_like(dct_coeffs)
        for i in range(self.n_bands):
            w = torch.sigmoid(self.band_weights[i]) * 2.0  # Scale to [0, 2]
            enhanced_dct = enhanced_dct + band_coeffs[i] * w

        # Step 4: Additional high-frequency refinement
        # Extract the original high-freq component for refinement
        hf_dct = band_coeffs[-1]  # Highest frequency band
        hf_image = idct_2d(hf_dct * band_masks[-1], norm='ortho')
        hf_refined = self.hf_enhance(hf_image)  # Learnable HF adjustment

        # Step 5: Apply body-part-dependent scaling
        if self.body_part_cond and body_part is not None:
            body_part_scale = self.body_part_to_scale(body_part)  # (B, 1)
            body_part_scale = body_part_scale.view(-1, 1, 1, 1)
            hf_refined = hf_refined * (1.0 + body_part_scale * 0.5)

        # Step 6: Reconstruct enhanced image
        enhanced_recon = idct_2d(enhanced_dct, norm='ortho')

        # Step 7: Add HF refinement as residual
        scale = 0.1  # Conservative HF boost
        output = enhanced_recon + scale * hf_refined

        return output

    def compute_frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-domain loss between prediction and target.
        Penalizes discrepancies in high-frequency DCT coefficients.

        Args:
            pred: (B, 1, H, W) predicted image
            target: (B, 1, H, W) target image

        Returns:
            freq_loss: scalar
        """
        # DCT of both
        pred_dct = dct_2d(pred, norm='ortho')
        target_dct = dct_2d(target, norm='ortho')

        # High-frequency mask (upper right quadrant)
        H, W = pred.shape[-2:]
        h_idx = torch.arange(H, device=pred.device)
        w_idx = torch.arange(W, device=pred.device)
        H_grid, W_grid = torch.meshgrid(h_idx, w_idx, indexing='ij')
        freq_dist = torch.sqrt(H_grid.float() ** 2 + W_grid.float() ** 2)
        hf_mask = (freq_dist > (H * 0.3)).float().view(1, 1, H, W)

        # Weighted L1 on high-frequency coefficients
        hf_diff = (pred_dct - target_dct).abs() * hf_mask
        freq_loss = hf_diff.mean()

        return freq_loss

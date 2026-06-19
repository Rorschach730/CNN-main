"""
Differentiable Radon Transform & Filtered Back-Projection (FBP)
================================================================
Implements sinogram ↔ image domain conversion using fully differentiable
PyTorch operations. Based on the standard PET reconstruction pipeline:
  - Forward: Image → Radon (sinogram) via ray-driven projection
  - Inverse: Sinogram → Image via ramp-filtered back-projection

Key features:
  - Fully differentiable (compatible with autograd)
  - FFT-based ramp filtering (Ram-Lak filter)
  - Torch grid_sample for efficient backprojection
  - Precomputed sampling grids (cached as buffers for speed)
  - Configurable number of projection views

v2.1: Grid caching — sampling grids precomputed once in __init__, eliminating
      Python for-loop overhead (180 iterations per forward pass).

Reference: Cross-Domain Reconstruction.pdf (Sinogram domain processing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RamLakFilter(nn.Module):
    """
    Ram-Lak (ramp) filter in frequency domain.
    H(ω) = |ω|, truncated at Nyquist frequency.
    Optionally applies a window (Hamming, Hann, Shepp-Logan) for noise suppression.
    """

    def __init__(self, n_bins: int, window: str = 'shepp-logan'):
        super().__init__()
        self.n_bins = n_bins

        freqs = torch.arange(0, n_bins, dtype=torch.float32)
        freqs = torch.min(freqs, n_bins - freqs)
        ramp = freqs / (n_bins / 2.0)

        if window == 'shepp-logan':
            eps = 1e-8
            sinc_val = torch.sin(math.pi * ramp + eps) / (math.pi * ramp + eps)
            window_vals = torch.where(ramp > eps, sinc_val, torch.ones_like(ramp))
            ramp = ramp * window_vals
        elif window == 'hamming':
            window_vals = 0.54 - 0.46 * torch.cos(2 * math.pi * freqs / n_bins)
            ramp = ramp * window_vals
        elif window == 'hann':
            window_vals = 0.5 * (1 - torch.cos(2 * math.pi * freqs / n_bins))
            ramp = ramp * window_vals

        self.register_buffer('ramp_filter', ramp.view(1, 1, -1))

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        B, C, n_views, n_bins = sinogram.shape
        sino_fft = torch.fft.rfft(sinogram, n=n_bins, dim=-1)
        ramp = self.ramp_filter[:, :, :sino_fft.shape[-1]]
        sino_fft_filtered = sino_fft * ramp
        filtered = torch.fft.irfft(sino_fft_filtered, n=n_bins, dim=-1)
        return filtered


class DifferentiableRadon(nn.Module):
    """
    Differentiable Radon transform for sinogram generation from 2D images.
    Uses bilinear interpolation-based ray integration.

    v2.1: Sampling grid precomputed once as buffer (was rebuilt per forward).

    Args:
        n_views: Number of projection angles
        img_size: Input image size (square)
        det_size: Number of detector bins (default: same as img_size)
    """

    def __init__(self, n_views: int = 256, img_size: int = 256, det_size: int = None):
        super().__init__()
        self.n_views = n_views
        self.img_size = img_size
        self.det_size = det_size or img_size

        angles = torch.linspace(0, math.pi, n_views, dtype=torch.float32)
        self.register_buffer('angles', angles)
        self.register_buffer('cos_angles', torch.cos(angles))
        self.register_buffer('sin_angles', torch.sin(angles))
        det_pos = torch.linspace(-1.0, 1.0, self.det_size, dtype=torch.float32)
        self.register_buffer('det_pos', det_pos)

        # ── v2.1: Precompute sampling grid once ──
        # Shape: (1, n_views, det_size, img_size, 2)
        # Stored without batch dim; expanded at forward time.
        self.register_buffer('cached_grid', self._build_grid())

    def _build_grid(self) -> torch.Tensor:
        """Build sampling grid for all views (called once in __init__)."""
        lin = torch.linspace(-1.0, 1.0, self.img_size)
        t_vals = self.det_pos      # (det_size,)
        s_vals = lin               # (img_size,)

        sampling_grids = []
        for i in range(self.n_views):
            cos_a = self.cos_angles[i]
            sin_a = self.sin_angles[i]

            T, S = torch.meshgrid(t_vals, s_vals, indexing='ij')
            x_coords = T * cos_a - S * sin_a   # (det_size, img_size)
            y_coords = T * sin_a + S * cos_a   # (det_size, img_size)
            grid = torch.stack([x_coords, y_coords], dim=-1)  # (det_size, img_size, 2)
            sampling_grids.append(grid)

        # Stack views: (n_views, det_size, img_size, 2)
        return torch.stack(sampling_grids, dim=0).unsqueeze(0)  # (1, n_views, det_size, img_size, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        n_views = self.n_views
        device = image.device

        # v2.1: Expand precomputed grid to batch size (moved to GPU once)
        grid = self.cached_grid.to(device=device).expand(B, -1, -1, -1, -1)
        # (B, n_views, det_size, img_size, 2)

        # Flatten batch+views for grid_sample
        grid_flat = grid.reshape(B * n_views, self.det_size, self.img_size, 2)

        # Expand image for each view
        image_expanded = image.unsqueeze(1).expand(-1, n_views, -1, -1, -1)
        image_expanded = image_expanded.reshape(B * n_views, C, H, W)

        sampled = F.grid_sample(
            image_expanded, grid_flat,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        sinogram_flat = sampled.sum(dim=-1)
        pixel_length = 2.0 / self.img_size
        sinogram_flat = sinogram_flat * pixel_length
        sinogram = sinogram_flat.reshape(B, C, n_views, self.det_size)
        return sinogram


class DifferentiableFBP(nn.Module):
    """
    Differentiable Filtered Back-Projection (FBP).
    Sinogram → Image reconstruction with ramp filtering + backprojection.

    v2.1: Sampling grid precomputed once as buffer (was rebuilt per forward
          with Python loop over n_views).

    Args:
        n_views: Number of projection angles
        img_size: Output image size
        det_size: Number of detector bins
        filter_window: Window function for ramp filter
    """

    def __init__(self, n_views: int = 256, img_size: int = 256,
                 det_size: int = None, filter_window: str = 'shepp-logan'):
        super().__init__()
        self.n_views = n_views
        self.img_size = img_size
        self.det_size = det_size or img_size

        self.ramp_filter = RamLakFilter(self.det_size, window=filter_window)

        angles = torch.linspace(0, math.pi, n_views, dtype=torch.float32)
        self.register_buffer('angles', angles)
        self.register_buffer('cos_angles', torch.cos(angles))
        self.register_buffer('sin_angles', torch.sin(angles))
        det_pos = torch.linspace(-1.0, 1.0, self.det_size, dtype=torch.float32)
        self.register_buffer('det_pos', det_pos)

        # ── v2.1: Precompute backprojection grid once ──
        # Shape: (1, n_views, H, W, 2)
        self.register_buffer('cached_grid', self._build_backproj_grid())

    def _build_backproj_grid(self) -> torch.Tensor:
        """Build backprojection grid for all views (called once in __init__)."""
        lin = torch.linspace(-1.0, 1.0, self.img_size)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')

        all_views = []
        for i in range(self.n_views):
            t_vals = xx * self.cos_angles[i] + yy * self.sin_angles[i]  # (H, W)
            grid_view = torch.stack([
                t_vals,
                torch.zeros_like(t_vals)
            ], dim=-1)  # (H, W, 2)
            all_views.append(grid_view)

        # (n_views, H, W, 2) → add batch dim: (1, n_views, H, W, 2)
        return torch.stack(all_views, dim=0).unsqueeze(0)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        B, C, n_views, det_size = sinogram.shape
        device = sinogram.device

        # Step 1: Ramp filtering
        sino_filtered = self.ramp_filter(sinogram)

        # Step 2: Backprojection using precomputed grid
        sino_flat = sino_filtered.permute(0, 2, 1, 3).contiguous()
        sino_flat = sino_flat.view(B * n_views, C, det_size)
        sino_flat = sino_flat.unsqueeze(2)  # (B*n_views, C, 1, det_size)

        # v2.1: Expand precomputed grid
        grids = self.cached_grid.to(device=device).expand(B, -1, -1, -1, -1)
        grids = grids.reshape(B * n_views, self.img_size, self.img_size, 2)

        backproj_pixels = F.grid_sample(
            sino_flat, grids,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        backproj_pixels = backproj_pixels.reshape(B, n_views, C, self.img_size, self.img_size)
        image = backproj_pixels.sum(dim=1)
        image = image * (math.pi / (2.0 * n_views))
        return image


class SinoImageBridge(nn.Module):
    """
    Complete sinogram ↔ image bridge for TriDo-CNN.
    Combines forward Radon + FBP into a single module.

    Forward path (training): Image → Sinogram → FBP → Image
    This creates a differentiable reconstruction pathway where gradients
    flow through both the sinogram and image domains.

    Args:
        n_views: Number of projection angles
        img_size: Image size
        det_size: Detector size (bin count)
        filter_window: Ramp filter window type
    """

    def __init__(self, n_views: int = 256, img_size: int = 256,
                 det_size: int = None, filter_window: str = 'shepp-logan'):
        super().__init__()
        self.radon = DifferentiableRadon(n_views, img_size, det_size)
        self.fbp = DifferentiableFBP(n_views, img_size, det_size, filter_window)

    def forward_project(self, image: torch.Tensor) -> torch.Tensor:
        """Image → Sinogram"""
        return self.radon(image)

    def reconstruct(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Sinogram → Image (FBP)"""
        return self.fbp(sinogram)

    def roundtrip(self, image: torch.Tensor) -> torch.Tensor:
        """Image → Sinogram → Image (consistency check)"""
        sino = self.radon(image)
        return self.fbp(sino)

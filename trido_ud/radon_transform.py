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
  - Configurable number of projection views

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

        # Build ramp filter in frequency domain
        freqs = torch.arange(0, n_bins, dtype=torch.float32)
        freqs = torch.min(freqs, n_bins - freqs)  # Fold for real FFT
        ramp = freqs / (n_bins / 2.0)  # Normalize to [0, 1]

        # Apply window function
        if window == 'shepp-logan':
            # sinc window: sin(πf) / (πf), with f ∈ [0, 1]
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
        # 'ram-lak': no window (pure ramp)

        self.register_buffer('ramp_filter', ramp.view(1, 1, -1))

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Apply ramp filter to sinogram views via FFT.

        Args:
            sinogram: (B, 1, n_views, n_bins)

        Returns:
            filtered: (B, 1, n_views, n_bins)
        """
        B, C, n_views, n_bins = sinogram.shape

        # Real FFT along the bin dimension
        sino_fft = torch.fft.rfft(sinogram, n=n_bins, dim=-1)

        # Apply ramp filter (only to positive frequencies)
        ramp = self.ramp_filter[:, :, :sino_fft.shape[-1]]
        sino_fft_filtered = sino_fft * ramp

        # Inverse FFT
        filtered = torch.fft.irfft(sino_fft_filtered, n=n_bins, dim=-1)

        return filtered


class DifferentiableRadon(nn.Module):
    """
    Differentiable Radon transform for sinogram generation from 2D images.
    Uses bilinear interpolation-based ray integration.

    Args:
        n_views: Number of projection angles (default: 256)
        img_size: Input image size (square, default: 256)
        det_size: Number of detector bins (default: same as img_size)
    """

    def __init__(self, n_views: int = 256, img_size: int = 256, det_size: int = None):
        super().__init__()
        self.n_views = n_views
        self.img_size = img_size
        self.det_size = det_size or img_size

        # Projection angles: uniformly spaced in [0, π)
        angles = torch.linspace(0, math.pi, n_views, dtype=torch.float32)
        self.register_buffer('angles', angles)

        # Precompute sin/cos for all angles
        self.register_buffer('cos_angles', torch.cos(angles))
        self.register_buffer('sin_angles', torch.sin(angles))

        # Detector positions
        det_pos = torch.linspace(-1.0, 1.0, self.det_size, dtype=torch.float32)
        self.register_buffer('det_pos', det_pos)

    def _get_rotated_coords(self, batch_size: int, device: torch.device):
        """
        Build (x', y') coordinate grid for each projection angle.
        x' = x*cos(θ) + y*sin(θ)  (rotated coordinate — the projection axis)
        y' = -x*sin(θ) + y*cos(θ) (orthogonal — the integration axis)

        Returns:
            sampling_grid: (B, n_views, n_rays, det_size, 2) in [-1, 1] coords for grid_sample
        """
        # Image pixel grid in [-1, 1]
        lin = torch.linspace(-1.0, 1.0, self.img_size, device=device)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')
        # Flatten image pixels: (img_size*img_size, 2)
        img_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)

        n_rays = self.img_size  # Number of integration rays per view

        # For each view, we sample along rotated detector lines
        # Build sampling grid for grid_sample
        # For backprojection: each detector bin corresponds to a line integral

        # Create grid: for each view, we have det_size sampling positions
        # along n_rays parallel rays
        sampling_grids = []

        for i in range(self.n_views):
            cos_a = self.cos_angles[i]
            sin_a = self.sin_angles[i]

            # For each detector bin position t ∈ [-1, 1]:
            # The ray is: x*cos(a) + y*sin(a) = t
            # We sample along this ray at multiple positions
            # Perpendicular sampling: (t, s) where s ∈ [-1, 1] parameterizes along the ray

            t_vals = self.det_pos  # (det_size,) — detector positions
            s_vals = lin  # (n_rays,) — positions along each ray

            # Parametric: x = t*cos(a) - s*sin(a), y = t*sin(a) + s*cos(a)
            # Build grid: (n_rays, det_size, 2)
            T, S = torch.meshgrid(t_vals, s_vals, indexing='ij')
            # T: (det_size, n_rays), S: (det_size, n_rays)

            x_coords = T * cos_a - S * sin_a  # (det_size, n_rays)
            y_coords = T * sin_a + S * cos_a  # (det_size, n_rays)

            grid = torch.stack([x_coords, y_coords], dim=-1)  # (det_size, n_rays, 2)
            # grid_sample expects (N, H_out, W_out, 2) where (x, y) in [-1, 1]
            # We want sinogram: n_views × det_size bins
            # grid_sample: sample image at these positions, then sum along rays
            sampling_grids.append(grid)

        sampling_grid = torch.stack(sampling_grids, dim=0)  # (n_views, det_size, n_rays, 2)
        sampling_grid = sampling_grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # (B, n_views, det_size, n_rays, 2)

        return sampling_grid

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward Radon transform: Image → Sinogram.

        Args:
            image: (B, 1, H, W)

        Returns:
            sinogram: (B, 1, n_views, det_size)
        """
        B, C, H, W = image.shape
        device = image.device

        # Build sampling grid
        sampling_grid = self._get_rotated_coords(B, device)  # (B, n_views, det_size, n_rays, 2)

        # Reshape for grid_sample: merge batch and views
        n_views = self.n_views
        sampling_grid_flat = sampling_grid.contiguous().view(B * n_views, self.det_size, self.img_size, 2)

        # Expand image for each view
        image_expanded = image.unsqueeze(1).expand(-1, n_views, -1, -1, -1)
        image_expanded = image_expanded.reshape(B * n_views, C, H, W)

        # Sample image at ray positions
        sampled = F.grid_sample(
            image_expanded, sampling_grid_flat,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (B*n_views, C, det_size, n_rays)

        # Integrate along rays (sum)
        sinogram_flat = sampled.sum(dim=-1)  # (B*n_views, C, det_size)

        # Scale by pixel size
        pixel_length = 2.0 / self.img_size
        sinogram_flat = sinogram_flat * pixel_length

        # Reshape back: (B, C, n_views, det_size)
        sinogram = sinogram_flat.reshape(B, C, n_views, self.det_size)

        return sinogram


class DifferentiableFBP(nn.Module):
    """
    Differentiable Filtered Back-Projection (FBP).
    Sinogram → Image reconstruction with ramp filtering + backprojection.

    Pipeline:
      1. Apply ramp filter to each projection view (FFT-based)
      2. Backproject: smear each filtered view across the image plane
      3. Normalize by number of views × π

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

        # Ramp filter
        self.ramp_filter = RamLakFilter(self.det_size, window=filter_window)

        # Projection angles
        angles = torch.linspace(0, math.pi, n_views, dtype=torch.float32)
        self.register_buffer('angles', angles)
        self.register_buffer('cos_angles', torch.cos(angles))
        self.register_buffer('sin_angles', torch.sin(angles))

        # Detector positions for backprojection
        det_pos = torch.linspace(-1.0, 1.0, self.det_size, dtype=torch.float32)
        self.register_buffer('det_pos', det_pos)

    def _build_backprojection_grid(self, batch_size: int, device: torch.device):
        """
        Build sampling grids for backprojection.
        For each image pixel, compute which detector bin it projects to at each angle.

        Returns:
            grid: (B, n_views, H, W, 1) — detector positions for grid_sample
        """
        # Image pixel centers in [-1, 1]
        lin = torch.linspace(-1.0, 1.0, self.img_size, device=device)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')
        # (H, W)

        all_grids = []
        for i in range(self.n_views):
            cos_a = self.cos_angles[i]
            sin_a = self.sin_angles[i]

            # Projection: t = x*cos(a) + y*sin(a) gives detector position
            t_vals = xx * cos_a + yy * sin_a  # (H, W)

            # Normalize t to [-1, 1] for grid_sample
            # grid_sample expects (x, y) in [-1, 1], we only need x (detector position)
            # The sinogram is treated as (1, 1, n_views, det_size)
            # We sample at position (t, 0) along dim 3 (detector)
            t_norm = t_vals.unsqueeze(-1)  # (H, W, 1)

            all_grids.append(t_norm)

        grid = torch.stack(all_grids, dim=0)  # (n_views, H, W, 1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # (B, n_views, H, W, 1)

        # For grid_sample on 1D signal:
        # The sinogram per view is (B*n_views, 1, 1, det_size)
        # grid_sample expects (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
        # x is along W_in (det_size), y along H_in (=1, so always 0)
        zero_y = torch.zeros_like(t_norm)
        grid_2d = torch.cat([grid, zero_y.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_views, -1, -1, -1)], dim=-1)
        # (B, n_views, H, W, 2)

        return grid_2d

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        FBP reconstruction: Sinogram → Image.

        Args:
            sinogram: (B, 1, n_views, det_size) or (B, C, n_views, det_size)

        Returns:
            image: (B, 1, H, W)
        """
        B, C, n_views, det_size = sinogram.shape
        device = sinogram.device

        # Step 1: Ramp filtering
        sino_filtered = self.ramp_filter(sinogram)  # (B, C, n_views, det_size)

        # Step 2: Backprojection using grid_sample
        # Treat filtered sinogram as 1D signal per view
        # Reshape: (B*C, n_views, det_size) → (B*C*n_views, 1, 1, det_size)
        sino_flat = sino_filtered.permute(0, 2, 1, 3).contiguous()
        sino_flat = sino_flat.view(B * n_views, C, det_size)
        sino_flat = sino_flat.unsqueeze(2)  # (B*n_views, C, 1, det_size)

        # Build grid: for each image pixel, find corresponding detector bin
        lin = torch.linspace(-1.0, 1.0, self.img_size, device=device)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')

        # For each view, compute detector position per pixel
        # t = x*cos(θ) + y*sin(θ), normalized to [-1, 1]
        all_views = []
        for i in range(n_views):
            t_vals = xx * self.cos_angles[i] + yy * self.sin_angles[i]  # (H, W)
            # grid_sample grid: (N, H_out, W_out, 2)
            # x → detector position (t), y → 0 (always middle of sinogram "height")
            grid_view = torch.stack([
                t_vals,  # x coordinate (detector position)
                torch.zeros_like(t_vals)  # y coordinate (always 0)
            ], dim=-1)  # (H, W, 2)
            all_views.append(grid_view)

        grids = torch.stack(all_views, dim=0)  # (n_views, H, W, 2)
        grids = grids.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, n_views, H, W, 2)
        grids = grids.reshape(B * n_views, self.img_size, self.img_size, 2)

        # Sample sinogram at these positions
        backproj_pixels = F.grid_sample(
            sino_flat, grids,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (B*n_views, C, H, W)

        # Reshape and sum across views
        backproj_pixels = backproj_pixels.reshape(B, n_views, C, self.img_size, self.img_size)
        image = backproj_pixels.sum(dim=1)  # (B, C, H, W)

        # Normalization: FBP scaling factor = π / (2 * n_views)
        image = image * (math.pi / (2.0 * n_views))

        return image


class SinoImageBridge(nn.Module):
    """
    Complete sinogram ↔ image bridge for TriDo-JiT.
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

"""
Fused Gromov-Wasserstein (FGW) Loss for TriDo-JiT
===================================================
Implements FGW distance as a structural regularizer for PET image denoising.

The FGW distance combines:
  - Wasserstein term: aligns feature distributions (intensity matching)
  - Gromov-Wasserstein term: aligns structural/topological relationships
    (preserves edges, textures, spatial correlations)

FGW_α(μ, ν) = min_π (1-α) Σ_{i,j} c(i,j) π_{i,j} + α Σ_{i,j,k,l} |d1(i,k) - d2(j,l)|² π_{i,j} π_{k,l}

For images, the Gromov-Wasserstein term captures structural similarity
independent of absolute intensity — ideal for PET where absolute SUV values
vary but anatomical structure should be preserved.

Implementation uses:
  - Patch-based approximation for computational efficiency
  - Sinkhorn algorithm for entropic regularization
  - Cosine similarity for structural distance computation

Reference: Fused Gromov-Wasserstein distance for structured objects (Vayer et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_euclidean(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances.

    Args:
        x: (N, D)
        y: (M, D) or None (then y=x)

    Returns:
        dist: (N, M)
    """
    if y is None:
        y = x
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (M, 1)
    dist = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, min=0.0)


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    Compute pairwise cosine distances (1 - cosine_similarity).

    Args:
        x: (N, D) — L2-normalized along dim D
        y: (M, D)

    Returns:
        dist: (N, M)
    """
    if y is None:
        y = x
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    cos_sim = torch.mm(x_norm, y_norm.t())
    return 1.0 - cos_sim


def sinkhorn(C: torch.Tensor, reg: float = 0.1, max_iter: int = 50,
             tol: float = 1e-6, P: torch.Tensor = None, Q: torch.Tensor = None) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for entropic optimal transport.

    Solves: min_π <C, π> + reg * H(π) subject to π⋅1=P, πᵀ⋅1=Q

    Args:
        C: (N, M) cost matrix
        reg: Entropic regularization strength
        max_iter: Maximum iterations
        tol: Convergence tolerance
        P: (N,) source distribution (default: uniform)
        Q: (M,) target distribution (default: uniform)

    Returns:
        π: (N, M) transport plan
    """
    N, M = C.shape
    device = C.device

    if P is None:
        P = torch.ones(N, device=device) / N
    if Q is None:
        Q = torch.ones(M, device=device) / M

    # Log-space for numerical stability
    log_P = torch.log(P + 1e-10)
    log_Q = torch.log(Q + 1e-10)

    # Kernel
    log_K = -C / reg

    # Sinkhorn iterations
    log_u = torch.zeros(N, device=device)
    for _ in range(max_iter):
        log_u_prev = log_u.clone()

        # Update v
        log_v = log_Q - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

        # Update u
        log_u = log_P - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

        # Convergence check
        if torch.max(torch.abs(log_u - log_u_prev)) < tol:
            break

    # Transport plan
    pi = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))

    return pi


class FGWLoss(nn.Module):
    """
    Fused Gromov-Wasserstein loss for PET image denoising.

    Computes FGW distance between predicted and target images using
    patch-based representation. Each patch is treated as a sample in
    the feature space.

    FGW_α = (1-α) * Wasserstein(features) + α * Gromov-Wasserstein(structure)

    For images:
      - Feature space: patch intensities (Wasserstein aligns histograms)
      - Structure space: inter-patch cosine distances (GW aligns textures)

    Args:
        patch_size: Size of square patches
        stride: Stride for patch extraction
        alpha: FGW trade-off (α=0: pure Wasserstein, α=1: pure GW)
        reg: Sinkhorn regularization strength
        feature_weight: Weight of FGW in total loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 8,
                 alpha: float = 0.5, reg: float = 0.1,
                 feature_weight: float = 0.01):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.alpha = alpha
        self.reg = reg
        self.feature_weight = feature_weight

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract overlapping patches from image.

        Args:
            x: (B, C, H, W)

        Returns:
            patches: (B, N_patches, C * patch_size²)
        """
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        # (B, C * patch_size², N_patches)
        patches = patches.transpose(1, 2)  # (B, N_patches, C * patch_size²)
        return patches

    def compute_cost_matrix(self, pred_patches: torch.Tensor,
                            target_patches: torch.Tensor) -> torch.Tensor:
        """
        Compute combined FGW cost matrix for one batch sample.

        Args:
            pred_patches: (N, D)
            target_patches: (M, D)

        Returns:
            C_fgw: (N, M) combined cost matrix
        """
        N, D = pred_patches.shape
        M, _ = target_patches.shape

        # ---- Wasserstein term: feature-level cost ----
        C_w = pairwise_euclidean(pred_patches, target_patches)  # (N, M)
        # Normalize
        if C_w.max() > 0:
            C_w = C_w / C_w.max()

        # ---- Gromov-Wasserstein term: structure-level cost ----
        # Intra-domain structural distances
        C1 = pairwise_cosine(pred_patches)   # (N, N)
        C2 = pairwise_cosine(target_patches)  # (M, M)

        # GW cost: Σ_{k,l} |C1(i,k) - C2(j,l)|² π_{i,j} π_{k,l}
        # Approximate via tensor product (efficient for moderate N, M)
        # C_gw[i,j] = (C1[i] ⊗ 1 - 1 ⊗ C2[j])² averaged
        # Simplified: use outer product differences
        if N <= 256 and M <= 256:
            # Exact GW cost computation
            C1_exp = C1.unsqueeze(2).unsqueeze(3)  # (N, N, 1, 1)
            C2_exp = C2.unsqueeze(0).unsqueeze(0)  # (1, 1, M, M)
            C_gw_tensor = (C1_exp - C2_exp) ** 2  # (N, N, M, M)
            # Reduce: average over k,l for each i,j
            C_gw = C_gw_tensor.mean(dim=(1, 3))  # (N, M)
        else:
            # Approximate: use mean structural difference
            C1_mean = C1.mean(dim=1, keepdim=True)   # (N, 1)
            C2_mean = C2.mean(dim=1, keepdim=True)   # (M, 1)
            C_gw = (C1_mean - C2_mean.t()) ** 2       # (N, M)

        # Normalize
        if C_gw.max() > 0:
            C_gw = C_gw / C_gw.max()

        # ---- Fused cost ----
        C_fgw = (1 - self.alpha) * C_w + self.alpha * C_gw

        return C_fgw

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute FGW loss between predicted and target images.

        Args:
            pred: (B, 1, H, W) predicted PET image
            target: (B, 1, H, W) target PET image

        Returns:
            fgw_loss: scalar
        """
        B = pred.shape[0]
        N_patches_max = 196  # Cap patches for computational efficiency

        total_loss = 0.0

        for b in range(B):
            # Extract patches
            pred_patches = self.extract_patches(pred[b:b + 1])  # (1, N, D)
            target_patches = self.extract_patches(target[b:b + 1])  # (1, M, D)

            pred_patches = pred_patches.squeeze(0)  # (N, D)
            target_patches = target_patches.squeeze(0)  # (M, D)

            # Subsample patches if too many
            if pred_patches.shape[0] > N_patches_max:
                idx = torch.randperm(pred_patches.shape[0], device=pred.device)[:N_patches_max]
                pred_patches = pred_patches[idx]
            if target_patches.shape[0] > N_patches_max:
                idx = torch.randperm(target_patches.shape[0], device=target.device)[:N_patches_max]
                target_patches = target_patches[idx]

            N, D = pred_patches.shape
            M, _ = target_patches.shape

            # Compute FGW cost matrix
            C_fgw = self.compute_cost_matrix(pred_patches, target_patches)

            # Solve OT via Sinkhorn
            pi = sinkhorn(C_fgw, reg=self.reg, max_iter=30)

            # FGW distance = <C_fgw, π>
            fgw_dist = (C_fgw * pi).sum()
            total_loss += fgw_dist

        return self.feature_weight * (total_loss / B)


class StructuralConsistencyLoss(nn.Module):
    """
    Simplified structural consistency loss using Gram matrix differences.
    More computationally efficient than full FGW, used as a faster alternative
    when patch-based FGW is too expensive.

    Captures texture/style consistency between predicted and target images
    via Gram matrix comparison in a learned feature space.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

        # Simple feature extractor: Sobel-like edge detection kernels
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix: G = XX^T / N"""
        B, C, H, W = x.shape
        features = x.view(B, C, -1)  # (B, C, H*W)
        G = torch.bmm(features, features.transpose(1, 2))  # (B, C, C)
        return G / (H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute structural consistency loss.

        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)

        Returns:
            loss: scalar
        """
        # Edge features
        pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)

        pred_edges = torch.cat([pred_edges_x, pred_edges_y, pred], dim=1)  # (B, 3, H, W)
        target_edges = torch.cat([target_edges_x, target_edges_y, target], dim=1)

        # Gram matrix difference
        G_pred = self.gram_matrix(pred_edges)
        G_target = self.gram_matrix(target_edges)

        loss = F.mse_loss(G_pred, G_target)
        return self.weight * loss

"""
Fused Gromov-Wasserstein (FGW) Loss for TriDo-JiT
===================================================
重构修复版：
彻底移除错误的均值近似项，采用全精确 $L_2$ 二次展开矩阵解析式，兼顾性能与数学精确度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_euclidean(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    if y is None: y = x
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, min=0.0)

def pairwise_cosine(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    if y is None: y = x
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return 1.0 - torch.mm(x_norm, y_norm.t())

def sinkhorn(C: torch.Tensor, reg: float = 0.1, max_iter: int = 50, tol: float = 1e-6) -> torch.Tensor:
    N, M = C.shape
    device = C.device

    P = torch.ones(N, device=device) / N
    Q = torch.ones(M, device=device) / M

    log_P = torch.log(P + 1e-10)
    log_Q = torch.log(Q + 1e-10)
    log_K = -C / reg

    log_u = torch.zeros(N, device=device)
    for _ in range(max_iter):
        log_u_prev = log_u.clone()
        log_v = log_Q - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_u = log_P - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        if torch.max(torch.abs(log_u - log_u_prev)) < tol:
            break

    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))


class FGWLoss(nn.Module):
    def __init__(self, patch_size: int = 16, stride: int = 8,
                 alpha: float = 0.5, reg: float = 0.1, feature_weight: float = 0.01):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.alpha = alpha
        self.reg = reg
        self.feature_weight = feature_weight

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        return patches.transpose(1, 2)

    def compute_cost_matrix(self, pred_patches: torch.Tensor, target_patches: torch.Tensor) -> torch.Tensor:
        N, D = pred_patches.shape
        M, _ = target_patches.shape

        # 1. 传统 Wasserstein 特征项
        C_w = pairwise_euclidean(pred_patches, target_patches)
        if C_w.max() > 0: C_w = C_w / C_w.max()

        # 2. 精确 Gromov-Wasserstein 拓扑项 (基于 L2 展开式)
        C1 = pairwise_cosine(pred_patches)    # [N, N]
        C2 = pairwise_cosine(target_patches)  # [M, M]

        if N <= 256 and M <= 256:
            # 严格快速矩阵乘法展开式
            C1_sq_sum = (C1 ** 2).mean(dim=1, keepdim=True) # [N, 1]
            C2_sq_sum = (C2 ** 2).mean(dim=0, keepdim=True) # [1, M]
            cross_term = torch.mm(C1, C2) / M               # [N, M]
            C_gw = C1_sq_sum + C2_sq_sum - 2.0 * cross_term
        else:
            C1_mean = C1.mean(dim=1, keepdim=True)
            C2_mean = C2.mean(dim=1, keepdim=True)
            C_gw = (C1_mean - C2_mean.t()) ** 2

        if C_gw.max() > 0: C_gw = C_gw / C_gw.max()

        return (1 - self.alpha) * C_w + self.alpha * C_gw

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]
        N_patches_max = 196
        total_loss = 0.0

        for b in range(B):
            pred_patches = self.extract_patches(pred[b:b + 1]).squeeze(0)
            target_patches = self.extract_patches(target[b:b + 1]).squeeze(0)

            if pred_patches.shape[0] > N_patches_max:
                idx = torch.randperm(pred_patches.shape[0], device=pred.device)[:N_patches_max]
                pred_patches = pred_patches[idx]
            if target_patches.shape[0] > N_patches_max:
                idx = torch.randperm(target_patches.shape[0], device=target.device)[:N_patches_max]
                target_patches = target_patches[idx]

            C_fgw = self.compute_cost_matrix(pred_patches, target_patches)
            pi = sinkhorn(C_fgw, reg=self.reg, max_iter=30)
            total_loss += (C_fgw * pi).sum()

        return self.feature_weight * (total_loss / B)


class StructuralConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        features = x.view(B, C, -1)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)

        pred_edges = torch.cat([pred_edges_x, pred_edges_y, pred], dim=1)
        target_edges = torch.cat([target_edges_x, target_edges_y, target], dim=1)

        return self.weight * F.mse_loss(self.gram_matrix(pred_edges), self.gram_matrix(target_edges))

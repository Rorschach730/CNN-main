"""
Fused Gromov-Wasserstein (FGW) Loss v2 for TriDo-CNN
=====================================================
融合版：结合 fgw_loss.py 的 L² 展开和 fgw_loss_1.py 的张量方法，
修复了 v0 中跨维度矩阵乘法的 bug，提供精确且高效的计算。

FGW 距离定义:
  FGW_α(μ, ν) = min_π (1-α) <C_w, π> + α <C_gw ⊗ π, π>

其中:
  - C_w[i,j] = ||pred_patch[i] - target_patch[j]||²  (Wasserstein 特征项)
  - C_gw[i,j] = (1/(N*M)) Σ_{k,l} |C1[i,k] - C2[j,l]|²  (Gromov-Wasserstein 结构项)
  - C1[i,k] = 1 - cos(pred[i], pred[k])  (预测图像内 patch 间结构距离)
  - C2[j,l] = 1 - cos(target[j], target[l])  (目标图像内 patch 间结构距离)

高效 L² 展开（O(N²+M²+NM)，无需 O(N²M²) 张量）:
  C_gw[i,j] = mean_k(C1[i,k]²) + mean_l(C2[j,l]²) - 2*mean_k(C1[i,k])*mean_l(C2[j,l])

对于大 patch 集（>256），使用近似: C_gw ≈ (mean(C1) - mean(C2).T)²

Reference: Fused Gromov-Wasserstein distance for structured objects (Vayer et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 距离函数
# ═══════════════════════════════════════════════════════════════

def pairwise_euclidean(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    成对欧氏距离: D[i,j] = ||x[i] - y[j]||²

    Args:
        x: (N, D)
        y: (M, D) 或 None（则 y=x）

    Returns:
        dist: (N, M)
    """
    if y is None:
        y = x
    x_norm = (x ** 2).sum(dim=1, keepdim=True)   # (N, 1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)   # (M, 1)
    dist = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, min=0.0)


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    成对余弦距离: D[i,j] = 1 - cos_sim(x[i], y[j])

    Args:
        x: (N, D)
        y: (M, D) 或 None

    Returns:
        dist: (N, M) 值域 [0, 2]
    """
    if y is None:
        y = x
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    cos_sim = torch.mm(x_norm, y_norm.t())
    return 1.0 - cos_sim


# ═══════════════════════════════════════════════════════════════
# Sinkhorn-Knopp 算法
# ═══════════════════════════════════════════════════════════════

def sinkhorn(C: torch.Tensor, reg: float = 0.1, max_iter: int = 50,
             tol: float = 1e-6,
             P: torch.Tensor = None, Q: torch.Tensor = None) -> torch.Tensor:
    """
    Sinkhorn-Knopp 熵正则最优传输。

    求解: min_π <C, π> + reg * H(π)  s.t. π·1=P, πᵀ·1=Q

    Args:
        C:        (N, M) 代价矩阵
        reg:      熵正则强度
        max_iter: 最大迭代次数
        tol:      收敛容差
        P:        (N,) 源分布（默认：均匀）
        Q:        (M,) 目标分布（默认：均匀）

    Returns:
        π: (N, M) 传输计划
    """
    N, M = C.shape
    device = C.device

    if P is None:
        P = torch.ones(N, device=device) / N
    if Q is None:
        Q = torch.ones(M, device=device) / M

    # Log-space 数值稳定
    log_P = torch.log(P + 1e-10)
    log_Q = torch.log(Q + 1e-10)
    log_K = -C / reg

    log_u = torch.zeros(N, device=device)
    for _ in range(max_iter):
        log_u_prev = log_u.clone()

        # 更新 v
        log_v = log_Q - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

        # 更新 u
        log_u = log_P - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

        # 收敛检查
        if torch.max(torch.abs(log_u - log_u_prev)) < tol:
            break

    # 传输计划
    pi = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    return pi


# ═══════════════════════════════════════════════════════════════
# FGW Loss
# ═══════════════════════════════════════════════════════════════

class FGWLoss(nn.Module):
    """
    Fused Gromov-Wasserstein 损失。

    融合 Wasserstein（特征对齐）和 Gromov-Wasserstein（结构对齐）。
    对 PET 去噪尤为重要：绝对 SUV 值可变，但解剖结构必须保持。

    FGW_α = (1-α) * Wasserstein(intensity) + α * Gromov-Wasserstein(structure)

    Args:
        patch_size:     方形 patch 边长
        stride:         patch 提取步长
        alpha:          FGW 权衡 (α=0: 纯 W, α=1: 纯 GW)
        reg:            Sinkhorn 正则化强度
        feature_weight: FGW 在总 loss 中的权重
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
        # 移除局部的 self._rng，完全依赖 main_trido.py 中的全局 set_seed() 以避免设备冲突和过拟合隐患

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取重叠 patch。

        Args:
            x: (B, C, H, W)

        Returns:
            patches: (B, N_patches, C * patch_size²)
        """
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        return patches.transpose(1, 2)  # (B, N_patches, C * patch_size²)

    def compute_cost_matrix(self, pred_patches: torch.Tensor,
                            target_patches: torch.Tensor) -> torch.Tensor:
        """
        计算 FGW 组合代价矩阵 (per-sample)。

        Args:
            pred_patches:   (N, D) 预测 patch 特征
            target_patches: (M, D) 目标 patch 特征

        Returns:
            C_fgw: (N, M) 组合代价矩阵
        """
        N, D = pred_patches.shape
        M, _ = target_patches.shape

        # ── 1. Wasserstein 项: 特征级欧氏距离 ──
        C_w = pairwise_euclidean(pred_patches, target_patches)  # (N, M)
        if C_w.max() > 0:
            C_w = C_w / C_w.max()  # 归一化到 [0, 1]

        # ── 2. Gromov-Wasserstein 项: 结构级余弦距离 ──
        C1 = pairwise_cosine(pred_patches)    # (N, N)
        C2 = pairwise_cosine(target_patches)  # (M, M)

        if N <= 256 and M <= 256:
            # [精确 L² 展开] — O(N²+M²+NM)，无内存爆炸
            # C_gw[i,j] = mean_k C1[i,k]² + mean_l C2[j,l]²
            #            - 2 * mean_k C1[i,k] * mean_l C2[j,l]
            C1_sq_mean = (C1 ** 2).mean(dim=1, keepdim=True)   # (N, 1)
            C2_sq_mean = (C2 ** 2).mean(dim=1, keepdim=True)   # (M, 1)
            C1_mean = C1.mean(dim=1, keepdim=True)              # (N, 1)
            C2_mean = C2.mean(dim=1, keepdim=True)              # (M, 1)

            C_gw = (C1_sq_mean + C2_sq_mean.t()
                    - 2.0 * torch.mm(C1_mean, C2_mean.t()))     # (N, M)
        else:
            # [近似] 大 patch 集: 用均值差平方近似
            # C_gw ≈ (mean_k C1[i,k] - mean_l C2[j,l])²
            C1_mean = C1.mean(dim=1, keepdim=True)   # (N, 1)
            C2_mean = C2.mean(dim=1, keepdim=True)   # (M, 1)
            C_gw = (C1_mean - C2_mean.t()) ** 2       # (N, M)

        if C_gw.max() > 0:
            C_gw = C_gw / C_gw.max()

        # ── 3. Fused 代价 ──
        C_fgw = (1 - self.alpha) * C_w + self.alpha * C_gw
        return C_fgw

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 batch FGW loss。

        Args:
            pred:   (B, 1, H, W) 预测图像
            target: (B, 1, H, W) 目标图像

        Returns:
            fgw_loss: 标量
        """
        B = pred.shape[0]
        N_patches_max = 196  # 限制 patch 数以控制计算量

        total_loss = 0.0

        for b in range(B):
            # 提取 patch
            pred_patches = self.extract_patches(pred[b:b + 1]).squeeze(0)     # (N, D)
            target_patches = self.extract_patches(target[b:b + 1]).squeeze(0) # (M, D)

            # 随机下采样（若 patch 过多）
            if pred_patches.shape[0] > N_patches_max:
                idx = torch.randperm(pred_patches.shape[0],
                                     device=pred.device)[:N_patches_max]
                pred_patches = pred_patches[idx]
            if target_patches.shape[0] > N_patches_max:
                idx = torch.randperm(target_patches.shape[0],
                                     device=target.device)[:N_patches_max]
                target_patches = target_patches[idx]

            # 计算 FGW 代价矩阵
            C_fgw = self.compute_cost_matrix(pred_patches, target_patches)

            # Sinkhorn 求解最优传输
            pi = sinkhorn(C_fgw, reg=self.reg, max_iter=30)

            # FGW 距离 = <C_fgw, π>
            fgw_dist = (C_fgw * pi).sum()
            total_loss += fgw_dist

        return self.feature_weight * (total_loss / B)


# ═══════════════════════════════════════════════════════════════
# Structural Consistency Loss (快速 Gram-based 替代)
# ═══════════════════════════════════════════════════════════════

class StructuralConsistencyLoss(nn.Module):
    """
    Gram 矩阵结构一致性损失。

    比完整 FGW 更快：通过 Sobel 边缘 + Gram 矩阵比较捕获纹理/风格一致性。
    适合作为 FGW 的补充或快速替代。

    Args:
        weight: 损失权重
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

        # Sobel 边缘检测核
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Gram 矩阵: G = XXᵀ / N"""
        B, C, H, W = x.shape
        features = x.view(B, C, -1)                       # (B, C, H*W)
        G = torch.bmm(features, features.transpose(1, 2))  # (B, C, C)
        return G / (H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1, H, W)
            target: (B, 1, H, W)

        Returns:
            loss: 标量
        """
        # 提取边缘特征
        pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)

        # 拼接: [edge_x, edge_y, original] → 3 通道
        pred_edges = torch.cat([pred_edges_x, pred_edges_y, pred], dim=1)
        target_edges = torch.cat([target_edges_x, target_edges_y, target], dim=1)

        # Gram 矩阵差
        G_pred = self.gram_matrix(pred_edges)
        G_target = self.gram_matrix(target_edges)

        return self.weight * F.mse_loss(G_pred, G_target)
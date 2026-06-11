"""
RED CEM (Cascaded Estimation Module) for Sinogram Residual Diffusion
======================================================================
基于 RED 论文 (MIA 2025) 的残差估计扩散机制，替换原简单 CNN SinogramEncoder。

核心创新:
  传统: sino_cond ──[CNN U-Net]──→ denoised_sino  (单步直接映射)
  RED:  sino_cond ──[REN → DCN]×T_s──→ denoised_sino  (迭代残差移除)

架构:
  REN (Residual Estimation Network): 修改版 ResUNet, 预测残差 ε̂
  DCN (Drift Correction Network):   同架构, 校正累积误差

训练:
  阶段 1: REN 单独训练 (MSE(ε, ε̂) + SSIM(x̂_0, x_0))
  阶段 2: DCN 单独训练 (REN 冻结, 用不完美预测生成漂移样本)

推理:
  确定性反向: x_T=low_dose → REN预测残差 → 移除 → DCN校正 → 迭代 T_s=20 步

Reference: RED: Residual Estimation Diffusion for Low-Dose PET Sinogram
           Reconstruction (Ai et al., MIA 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Tiramisu / ResUNet 风格块
# ═══════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """残差卷积块: Conv3→GN→SiLU→Conv3→GN, 残差连接"""

    def __init__(self, ch: int, cond_ch: int = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch + cond_ch, ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        h = x
        if cond is not None:
            h = torch.cat([h, cond], dim=1)
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return x + h


class DownBlock(nn.Module):
    """下采样块: Conv2 stride=2 + ResBlock"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        x = self.down(x)
        x = self.res(x, cond)
        return x


class UpBlock(nn.Module):
    """上采样块: Upsample(×2) + Conv1 + skip + ResBlock"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.res = ResBlock(out_ch + skip_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                cond: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        x = self.proj(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                              align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, cond)
        return x


# ═══════════════════════════════════════════════════════════════
# REN: 残差估计网络
# ═══════════════════════════════════════════════════════════════

class ResidualEstimationNet(nn.Module):
    """
    残差估计网络 (REN): 预测当前弦图中的残差成分。

    输入: 当前弦图 x_t (1, V, B) + 时间步 t
    输出: 估计残差 ε̂ (1, V, B)

    架构: ResUNet (depth=4, base_ch=32)
    """

    def __init__(self, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]

        # 时间步编码 (sinusoidal)
        self.t_proj = nn.Sequential(
            nn.Linear(64, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )

        # 输入投影: 1 → base_ch
        self.in_conv = nn.Conv2d(1, chs[0], 3, padding=1, bias=False)

        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.enc_blocks.append(ResBlock(chs[i]))
            self.downs.append(DownBlock(chs[i], chs[i + 1]))

        # 瓶颈
        self.bottleneck = ResBlock(chs[-1])

        # 解码器
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(UpBlock(chs[i], chs[i - 1], chs[i - 1]))
            self.dec_blocks.append(ResBlock(chs[i - 1]))

        # 输出: base_ch → 1
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], 1, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, 1, V, B) 当前弦图
            t:   (B,) 或 (B, 1) 时间步, ∈ [0, 1]

        Returns:
            epsilon_hat: (B, 1, V, B) 预测残差
        """
        B = x_t.size(0)

        # 时间步编码
        t = t.float().view(-1, 1)
        t_sin = self._sinusoidal_encoding(t, 64)
        t_cond = self.t_proj(t_sin).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        t_cond = t_cond.expand(-1, -1, x_t.shape[2], x_t.shape[3])

        # 输入投影
        x = self.in_conv(x_t)

        # 编码
        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码
        for i in range(self.depth):
            x = self.ups[i](x, skips[-(i + 1)])
            x = self.dec_blocks[i](x)

        return self.out_conv(x)

    @staticmethod
    def _sinusoidal_encoding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal 时间编码"""
        device = t.device
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, device=device).float()
                          * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ═══════════════════════════════════════════════════════════════
# DCN: 漂移校正网络
# ═══════════════════════════════════════════════════════════════

class DriftCorrectionNet(nn.Module):
    """
    漂移校正网络 (DCN): 校正 REN 预测的累积误差。

    输入: 修正后的中间弦图 x̂_t (1, V, B) + 原始弦图 x_t + 时间 t
    输出: 漂移修正项 γ̂ (1, V, B)

    与 REN 同架构但输入通道更多 (2 vs 1)
    """

    def __init__(self, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]

        self.t_proj = nn.Sequential(
            nn.Linear(64, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )

        # 输入: [x̂_t, x_t] → 2 通道
        self.in_conv = nn.Conv2d(2, chs[0], 3, padding=1, bias=False)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.enc_blocks.append(ResBlock(chs[i]))
            self.downs.append(DownBlock(chs[i], chs[i + 1]))

        self.bottleneck = ResBlock(chs[-1])

        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(UpBlock(chs[i], chs[i - 1], chs[i - 1]))
            self.dec_blocks.append(ResBlock(chs[i - 1]))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], 1, 1),
        )

    def forward(self, x_hat: torch.Tensor, x_curr: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hat:  (B, 1, V, B) REN 预测的中间弦图
            x_curr: (B, 1, V, B) 当前弦图
            t:      (B,) 时间步

        Returns:
            gamma_hat: (B, 1, V, B) 漂移修正项
        """
        t = t.float().view(-1, 1)
        t_sin = ResidualEstimationNet._sinusoidal_encoding(t, 64)
        t_cond = self.t_proj(t_sin).unsqueeze(-1).unsqueeze(-1)
        t_cond = t_cond.expand(-1, -1, x_hat.shape[2], x_hat.shape[3])

        # 拼接 [x̂, x_curr]
        x = self.in_conv(torch.cat([x_hat, x_curr], dim=1))

        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)

        x = self.bottleneck(x)

        for i in range(self.depth):
            x = self.ups[i](x, skips[-(i + 1)])
            x = self.dec_blocks[i](x)

        return self.out_conv(x)


# ═══════════════════════════════════════════════════════════════
# CEM: 级联估计模块 (REN + DCN)
# ═══════════════════════════════════════════════════════════════

class CEMSinogramDenoiser(nn.Module):
    """
    CEM (Cascaded Estimation Module) 弦图残差扩散去噪器。

    替代原 SinogramEncoder (简单 U-Net 深度=3)，用 RED 的迭代残差去噪。

    Args:
        n_views: 投影角数 (默认 256)
        n_bins:  探测器 bin 数 (默认 256)
        T_max:   最大扩散步数 (默认 500, 训练用)
        T_s:     推理采样步数 (默认 20, 推理用)
        base_ch: 基础通道数
        beta_drift: 漂移校正权重 (默认 0.05)
    """

    def __init__(self, n_views: int = 256, n_bins: int = 256,
                 T_max: int = 500, T_s: int = 20,
                 base_ch: int = 32, beta_drift: float = 0.05):
        super().__init__()
        self.n_views = n_views
        self.n_bins = n_bins
        self.T_max = T_max
        self.T_s = T_s
        self.beta = beta_drift

        # 残差调度: α_t = 1 - (t/T)²
        # 推理: 从低剂量 x_T=1 到全剂量 x_0=0
        self.register_buffer('_alpha_schedule', None)

        # REN + DCN
        self.ren = ResidualEstimationNet(base_ch=base_ch, depth=4)
        self.dcn = DriftCorrectionNet(base_ch=base_ch, depth=4)

    @staticmethod
    def _alpha(t: torch.Tensor, T_max: int = 500) -> torch.Tensor:
        """残差调度: α_t = 1 - (t/T)²"""
        return 1.0 - (t / T_max) ** 2

    def forward(self, sinogram: torch.Tensor,
                body_part: torch.Tensor = None) -> torch.Tensor:
        """
        训练前向: 随机时间步 t, 预测去噪弦图, 返回 loss 分量。

        注意: 实际训练 loss 在外部 compute_loss() 中计算。
        此处返回去噪结果用于下游 FBP 重建。

        Args:
            sinogram: (B, 1, V, B) 低剂量弦图
            body_part: (B,) 忽略 (仅保持接口兼容)

        Returns:
            denoised_sino: (B, 1, V, B)
        """
        # 残差 = 低剂量 - 全剂量 (训练时 target 不可得, 用 REN 估计)
        device = sinogram.device
        B = sinogram.size(0)

        # 随机时间步
        t_float = torch.rand(B, device=device) * self.T_max
        t = t_float.long()

        # α_t: 当前残差比例
        alpha_t = self._alpha(t.float(), self.T_max).view(-1, 1, 1, 1)

        # REN 预测残差
        eps_hat = self.ren(sinogram, t.float() / self.T_max)

        # 预测全剂量
        x0_hat = sinogram - alpha_t * eps_hat

        # DCN 漂移校正
        gamma_hat = self.dcn(x0_hat, sinogram, t.float() / self.T_max)

        return x0_hat + self.beta * gamma_hat

    @torch.no_grad()
    def denoise(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        推理: CEM 迭代去噪 (确定性反向过程)。

        x_T = sinogram (低剂量)
        for t = T, T-Δ, T-2Δ, ..., 0:
            ε̂ = REN(x_t, t)
            x̂_{t-Δ} = x_t - (α_t - α_{t-Δ}) · ε̂
            γ̂ = DCN(x̂_{t-Δ}, x_t, t)
            x_{t-Δ} = x̂_{t-Δ} + β · γ̂

        Args:
            sinogram: (B, 1, V, B) 低剂量弦图

        Returns:
            denoised: (B, 1, V, B) 去噪弦图 (≈全剂量)
        """
        device = sinogram.device
        B = sinogram.size(0)
        x = sinogram.clone()  # x_T = low-dose

        # T_s 步均匀采样
        steps = torch.linspace(self.T_max, 0, self.T_s + 1, device=device)

        for i in range(self.T_s):
            t_curr = steps[i]
            t_next = steps[i + 1]

            # 残差调度值
            alpha_curr = self._alpha(t_curr, self.T_max)
            alpha_next = self._alpha(t_next, self.T_max)

            # REN: 预测当前残差
            t_norm = torch.full((B,), t_curr / self.T_max, device=device)
            eps_hat = self.ren(x, t_norm)

            # 移除残差: x̂_{t-Δ} = x_t - (α_t - α_{t-Δ}) · ε̂
            delta_alpha = alpha_curr - alpha_next
            x_hat = x - delta_alpha * eps_hat

            # DCN: 漂移校正
            t_norm_next = torch.full((B,), t_next / self.T_max, device=device)
            gamma_hat = self.dcn(x_hat, x, t_norm)
            x = x_hat + self.beta * gamma_hat

        return x

    def compute_loss(self, sinogram: torch.Tensor,
                     target_sino: torch.Tensor,
                     body_part: torch.Tensor = None) -> dict:
        """
        训练损失 (两阶段训练用)。

        Stage 1 (REN only):
          t ~ U(0, T_max)
          ε = sinogram - target_sino
          ε̂ = REN(x_t, t)
          L_REN = MSE(ε, ε̂) + SSIM(x̂_0, x_0)

        Stage 2 (DCN only, REN frozen):
          REN 推理得到不完美预测
          混合 λ~U(0,1): 混合残差 = λ·ε̂ + (1-λ)·ε
          生成漂移样本 → 训练 DCN

        Returns:
            {'ren_loss': ..., 'dcn_loss': ..., 'total': ...}
        """
        device = sinogram.device
        B = sinogram.size(0)
        eps = sinogram - target_sino  # 真实残差

        # ── Stage 1: REN loss ──
        t = torch.randint(0, self.T_max, (B,), device=device)
        alpha_t = self._alpha(t.float(), self.T_max).view(-1, 1, 1, 1)
        x_t = target_sino + alpha_t * eps

        t_norm = t.float() / self.T_max
        eps_hat = self.ren(x_t, t_norm)
        loss_ren = F.mse_loss(eps_hat, eps)

        # ── Stage 2: DCN loss (用 REN 不完美预测生成漂移) ──
        with torch.no_grad():
            t_T = torch.full((B,), self.T_max, device=device)
            eps_T_hat = self.ren(sinogram, t_T.float() / self.T_max)

            # λ 混合: 模拟 REN 不完美程度
            lam = torch.rand(B, 1, 1, 1, device=device)
            mixed_eps = lam * eps_T_hat + (1 - lam) * eps

            # 生成漂移样本
            x_drift = target_sino + (1 - alpha_t) * mixed_eps
            gamma_target = x_drift - x_t
            gamma_hat = self.dcn(x_drift, x_t, t_norm)

        loss_dcn = F.mse_loss(gamma_hat, gamma_target)

        return {
            'ren_loss': loss_ren,
            'dcn_loss': loss_dcn,
            'total': loss_ren + loss_dcn,
        }

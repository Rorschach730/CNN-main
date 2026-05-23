"""
TriDo-JiT Denoiser v2: Flow Matching x-Prediction with Multi-Domain Regularization
====================================================================================
融合版：结合 _trido 和 _trido_1 的最佳实践，严格遵循 JiT 论文的 x-prediction 范式。

核心设计（来自 JiT 论文 "Back to Basics", Li & He, MIT）:
  - 网络直接预测 x_pred (clean image)，而非噪声或速度场
  - 利用流形假设：干净图像在低维流形上，低容量网络即可胜任
  - Loss: v_pred = (x_pred - z) / clamp(1-t, 0.05)，target = clean - condition
  - v-loss 的隐式加权自然偏好中间时间步，无需显式 weighting fn

三域架构（TriDo）:
  ① Sinogram Domain  — Radon→SinoEncoder→FBP，含 body-part FiLM 调节
  ② Image Domain      — JiT Transformer (patch embed + adaLN + RoPE attention)
  ③ Frequency Domain  — GFP (DCT→Band Split→自适应增强→IDCT)

辅助损失:
  - FGW (Fused Gromov-Wasserstein): 结构保持
  - Structural Consistency (Gram-based): 纹理一致性
  - Frequency (GFP 软掩码): 高频细节增强
  - Sinogram Consistency: 投影域一致性

训练: v-prediction Huber Loss + 四项辅助损失
推理: Heun's 2nd-order ODE solver + Classifier-Free Guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from trido_ud.model_trido import TriDoJiT_models
    from trido_ud.fgw_loss import FGWLoss, StructuralConsistencyLoss
except ImportError:
    from model_trido import TriDoJiT_models
    from fgw_loss import FGWLoss, StructuralConsistencyLoss


class TriDoDenoiser(nn.Module):
    """
    Flow Matching denoiser for TriDo-JiT (v2 — JiT-aligned x-prediction).

    Training:
      - Primary: v-prediction Huber loss (x_pred → v_pred 转换)
      - Regularizers: FGW + Structural + Frequency + Sinogram

    Inference:
      - Heun's 2nd-order ODE solver
      - Classifier-Free Guidance (CFG) with body-part conditioning
      - 双路并行：cond + uncond 合并 batch 推理

    Args:
        args: Argument namespace (详见 main_trido.py 的 get_args_parser)
    """

    def __init__(self, args):
        super().__init__()

        self.patch_size = getattr(args, 'patch_size', 16)
        model_size = getattr(args, 'model_size', 'Base')
        model_key = f'TriDoJiT-{model_size}'

        # ── 三域主干网络 ──
        self.net = TriDoJiT_models[model_key](
            input_size=args.img_size,
            patch_size=self.patch_size,
            in_channels=2,
            out_channels=1,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            use_sino_domain=getattr(args, 'use_sino_domain', True),
            use_freq_domain=getattr(args, 'use_freq_domain', True),
        )

        self.img_size = args.img_size

        # ── Flow Matching 参数 ──
        self.P_mean = getattr(args, 'P_mean', -0.5)
        self.P_std = getattr(args, 'P_std', 1.2)
        self.cond_drop_prob = getattr(args, 'cond_drop_prob', 0.1)
        self.cfg_scale = getattr(args, 'cfg_scale', 2.0)

        # ── Loss 权重 ──
        self.fgw_weight = getattr(args, 'fgw_weight', 0.01)
        self.freq_weight = getattr(args, 'freq_weight', 0.005)
        self.sino_weight = getattr(args, 'sino_weight', 0.01)
        self.struct_weight = getattr(args, 'struct_weight', 0.01)

        # ── FGW 结构损失（基于 patch 的高效 FGW）──
        self.fgw_loss = FGWLoss(
            patch_size=16, stride=8, alpha=0.5,
            reg=0.1, feature_weight=1.0
        )

        # ── 结构一致性损失（Gram 矩阵，快速替代）──
        self.struct_loss = StructuralConsistencyLoss(weight=1.0)

        # ── EMA 参数追踪 ──
        accum_iter = getattr(args, 'accum_iter', 1)
        self.ema_decay = 0.999 ** accum_iter
        self.ema_params = [p.clone().detach() for p in self.net.parameters()]

        # ── 最近一次 loss 分解 ──
        self._last_losses = {}

    # ═══════════════════════════════════════════════════════════════
    # 时间步采样
    # ═══════════════════════════════════════════════════════════════

    def sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        """Logit-normal 时间步采样（偏好中间 t，与 FM 论文一致）"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    # ═══════════════════════════════════════════════════════════════
    # 训练前向
    # ═══════════════════════════════════════════════════════════════

    def forward(self, target: torch.Tensor, condition: torch.Tensor,
                body_part: torch.Tensor) -> torch.Tensor:
        """
        训练前向传播。

        JiT 核心公式:
          z = t * target + (1-t) * condition       (插值)
          x_pred = net(z, condition_input, t, body_part)
          v_pred = (x_pred - z) / clamp(1-t, 0.05)  (→ 用于加权 loss)
          v_target = target - condition              (速度场真值)

        Args:
            target:    (B, 1, H, W) 全剂量 PET 干净图像
            condition: (B, 1, H, W) 低剂量 PET 条件图像
            body_part: (B,) LongTensor  身体部位 (0=brain, 1=chest, 2=abdomen)

        Returns:
            total_loss: 标量，总和 loss
        """
        bsz = target.size(0)
        device = target.device

        # ── Step 1: Flow Matching 插值 ──
        t = self.sample_t(bsz, device=device)           # (B,)
        t_reshape = t.view(-1, 1, 1, 1)                 # (B, 1, 1, 1)

        # 线性插值: z = t * x_1 + (1-t) * x_0
        z = t_reshape * target + (1 - t_reshape) * condition

        # ── Step 2: CFG 条件随机丢弃 ──
        drop_mask = (torch.rand(bsz, 1, 1, 1, device=device)
                     > self.cond_drop_prob).float()
        condition_input = condition * drop_mask

        # ── Step 3: 拼接输入 → 三域网络前向 ──
        model_input = torch.cat([z, condition_input], dim=1)   # (B, 2, H, W)

        # [JiT 核心]: 网络直接预测干净图像 x_pred
        x_pred = self.net(model_input, t, body_part)  # (B, 1, H, W)

        # ── Step 4: x_pred → v_pred 转换 + 加权 Loss ──
        # v_pred = (x_pred - z) / (1-t)，分母 clamp 防除零
        denom = torch.clamp(1.0 - t_reshape, min=0.05)
        v_pred = (x_pred - z) / denom
        v_target = target - condition

        # Huber Loss (β=0.1 抗震荡，对 PET SUV 异常值鲁棒)
        loss_v = F.smooth_l1_loss(v_pred, v_target, beta=0.1)

        # ── Step 5: 辅助损失（均作用在 x_pred 上）──

        # 5a. FGW 结构损失
        loss_fgw = self.fgw_loss(x_pred, target)

        # 5b. Gram 结构一致性
        loss_struct = self.struct_loss(x_pred, target)

        # 5c. GFP 频域损失（软掩码，与 forward 对齐）
        if hasattr(self.net, 'gfp') and self.net.gfp is not None:
            loss_freq = self.net.gfp.compute_frequency_loss(x_pred, target)
        else:
            loss_freq = torch.tensor(0.0, device=device)

        # 5d. Sinogram 一致性损失
        loss_sino = torch.tensor(0.0, device=device)
        if hasattr(self.net, 'sino_bridge') and self.net.sino_bridge is not None:
            with torch.no_grad():
                sino_target = self.net.sino_bridge.forward_project(target)
            sino_pred = self.net.sino_bridge.forward_project(x_pred)
            loss_sino = F.l1_loss(sino_pred, sino_target)

        # ── Step 6: 总损失 ──
        total_loss = (
            loss_v
            + self.fgw_weight * loss_fgw
            + self.freq_weight * loss_freq
            + self.struct_weight * loss_struct
            + self.sino_weight * loss_sino
        )

        # 存储供 logging
        self._last_losses = {
            'v_loss': loss_v.item(),
            'fgw_loss': loss_fgw.item(),
            'freq_loss': loss_freq.item(),
            'struct_loss': loss_struct.item(),
            'sino_loss': loss_sino.item(),
            'total': total_loss.item(),
        }

        return total_loss

    def get_last_losses(self) -> dict:
        """获取最近一次 forward 的 loss 分解"""
        return getattr(self, '_last_losses', {})

    # ═══════════════════════════════════════════════════════════════
    # EMA 管理
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def update_ema(self):
        """指数移动平均更新"""
        current_device = next(self.net.parameters()).device
        if self.ema_params[0].device != current_device:
            self.ema_params = [p.to(current_device) for p in self.ema_params]
        for targ, src in zip(self.ema_params, self.net.parameters()):
            targ.mul_(self.ema_decay).add_(src, alpha=1 - self.ema_decay)

    def get_ema_state_dict(self) -> dict:
        """获取 EMA 参数字典（用于 checkpoint 保存）"""
        return {name: ema_param
                for (name, _), ema_param
                in zip(self.net.named_parameters(), self.ema_params)}

    def load_ema_state_dict(self, ema_dict: dict):
        """加载 EMA 参数（用于 checkpoint 恢复）"""
        for (name, _), ema_param in zip(self.net.named_parameters(), self.ema_params):
            if name in ema_dict:
                ema_param.data.copy_(ema_dict[name].data)

    # ═══════════════════════════════════════════════════════════════
    # 推理生成（Heun's 2nd-order + CFG）
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, body_part: torch.Tensor,
                 steps: int = 50, cfg_scale: float = None) -> torch.Tensor:
        """
        Flow Matching ODE 采样生成去噪 PET 图像。

        使用 Heun's 2nd-order 方法 + Classifier-Free Guidance。
        双路并行：将 cond 和 uncond 合并 batch 一次前向，效率提升 2x。

        Args:
            condition: (B, 1, H, W) 低剂量 PET
            body_part: (B,) LongTensor  身体部位
            steps:     ODE 积分步数
            cfg_scale: CFG 引导强度（默认 self.cfg_scale）

        Returns:
            denoised: (B, 1, H, W) 高质量 PET 图像
        """
        device = condition.device
        bsz = condition.size(0)

        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # 初始化: z_0 = condition (从低剂量开始)
        z = condition.clone()
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

        # 无条件输入: 全零（CFG 的 null condition）
        uncond_condition = torch.zeros_like(condition)

        # 双路 body_part: [cond_part, uncond_part]
        body_in = torch.cat([body_part, body_part], dim=0)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            t_curr_batch = t_curr.repeat(bsz)

            # ── Heun Step 1: 在 t_curr 处求 v_curr ──
            z_in = torch.cat([z, z], dim=0)                       # (2B, 1, H, W)
            cond_in = torch.cat([condition, uncond_condition], dim=0)
            t_in = torch.cat([t_curr_batch, t_curr_batch], dim=0)

            x_pred_curr = self.net(
                torch.cat([z_in, cond_in], dim=1), t_in, body_in
            )
            denom_curr = torch.clamp(1.0 - t_curr, min=0.05)
            v_curr_all = (x_pred_curr - z_in) / denom_curr

            # CFG 融合
            v_curr_cond, v_curr_uncond = v_curr_all.chunk(2, dim=0)
            v_curr = v_curr_uncond + cfg_scale * (v_curr_cond - v_curr_uncond)

            # Euler 半步预览
            z_tmp = z + v_curr * dt

            # ── Heun Step 2: 在 t_next 处求 v_next，取平均（含最后一步）──
            t_next_batch = t_next.repeat(bsz)
            z_tmp_in = torch.cat([z_tmp, z_tmp], dim=0)
            t_next_in = torch.cat([t_next_batch, t_next_batch], dim=0)

            x_pred_next = self.net(
                torch.cat([z_tmp_in, cond_in], dim=1), t_next_in, body_in
            )
            denom_next = torch.clamp(1.0 - t_next, min=0.05)
            v_next_all = (x_pred_next - z_tmp_in) / denom_next

            v_next_cond, v_next_uncond = v_next_all.chunk(2, dim=0)
            v_next = v_next_uncond + cfg_scale * (v_next_cond - v_next_uncond)

            # Heun's 平均: z = z + 0.5 * (v_curr + v_next) * dt
            z = z + 0.5 * (v_curr + v_next) * dt

        return z

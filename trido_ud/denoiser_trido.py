"""
TriDo-CNN Denoiser v3: Flow Matching x-Prediction with Multi-Domain Regularization
====================================================================================
融合版：结合 _trido 和 _trido_1 的最佳实践，严格遵循 Flow Matching 论文的 x-prediction 范式。

核心设计（基于 Flow Matching "Back to Basics" 范式, Lipman et al.）:
  - 网络直接预测 x_pred (clean image)，而非噪声或速度场
  - 利用流形假设：干净图像在低维流形上，低容量网络即可胜任
  - Loss: v_pred = (x_pred - z) / clamp(1-t, 0.05)，target = clean - condition
  - v-loss 的隐式加权自然偏好中间时间步，无需显式 weighting fn

三域架构（TriDo-CNN）:
  ① Sinogram Domain  — Radon→SinoEncoder→FBP，含 body-part FiLM 调节
  ② Image Domain      — ResNet U-Net CNN (FiLM 条件注入 + 跳跃连接)
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
    from trido_ud.model_cnn import TriDoCNN_models
    from trido_ud.fgw_loss import FGWLoss, StructuralConsistencyLoss
except ImportError:
    from model_cnn import TriDoCNN_models
    from fgw_loss import FGWLoss, StructuralConsistencyLoss


class TriDoDenoiser(nn.Module):
    """
    Flow Matching denoiser for TriDo-CNN (v3 — CNN-based x-prediction).

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
        model_key = f'TriDoCNN-{model_size}'

        # ── 三域主干网络 ──
        self.net = TriDoCNN_models[model_key](
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

        # 5c. GFP 频域损失（软掩码 DCT，与 forward 对齐）
        if hasattr(self.net, 'gfp') and self.net.gfp is not None:
            loss_freq = self.net.gfp.compute_frequency_loss(x_pred, target)
        else:
            loss_freq = torch.tensor(0.0, device=device)

        # 5d. HALO 复合频域损失（全局 FFT + 局部 DWT 双重约束）
        loss_freq_halo = self._compute_halo_frequency_loss(x_pred, target)

        # 5e. Sinogram 一致性损失
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
            + self.freq_weight * 2.0 * loss_freq_halo  # HALO: 2× FFT, 1× DWT
            + self.struct_weight * loss_struct
            + self.sino_weight * loss_sino
        )

        # 存储供 logging
        self._last_losses = {
            'v_loss': loss_v.item(),
            'fgw_loss': loss_fgw.item(),
            'freq_loss': loss_freq.item(),
            'freq_halo_loss': loss_freq_halo.item(),
            'struct_loss': loss_struct.item(),
            'sino_loss': loss_sino.item(),
            'total': total_loss.item(),
        }

        return total_loss

    def get_last_losses(self) -> dict:
        """获取最近一次 forward 的 loss 分解"""
        return getattr(self, '_last_losses', {})

    # ═══════════════════════════════════════════════════════════════
    # HALO 复合频域损失（全局 FFT + 局部 DWT）
    # ═══════════════════════════════════════════════════════════════

    def _compute_halo_frequency_loss(self, pred: torch.Tensor,
                                     target: torch.Tensor) -> torch.Tensor:
        """
        HALO-style 复合频域损失:
          L_freq = 0.2·L_FFT + 0.1·L_DWT
        """
        # 强制转换为 float32，因为 FFT 不支持 bfloat16
        pred = pred.float()
        target = target.float()

        # ── 全局 FFT 频谱一致性 ──
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        loss_fft = F.l1_loss(pred_fft.real, target_fft.real) \
                   + F.l1_loss(pred_fft.imag, target_fft.imag)

        # ── 局部 DWT 高频子带细节 ──
        loss_dwt = self._compute_dwt_hf_loss(pred, target)

        return 0.2 * loss_fft + 0.1 * loss_dwt

    def _compute_dwt_hf_loss(self, pred: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """
        Haar DWT 两级分解，对 LH/HL/HH 高频子带做 L1 约束。

        DWT 将图像分解为:
          Level 1: [LL, LH, HL, HH]  (各 H/2 × W/2)
          Level 2: [LL2, LH2, HL2, HH2]  (各 H/4 × W/4)

        只对高频子带 (LH/HL/HH) 做 L1 loss，不约束低频 LL。
        """
        def haar_dwt_2d(x):
            """Haar 2D DWT: 返回 [LL, LH, HL, HH]"""
            B, C, H, W = x.shape
            # Low-pass (平均) 和 High-pass (差分)
            L = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) / 2.0  # 水平低通
            H = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) / 2.0  # 水平高通
            LL = (L[:, :, 0::2, :] + L[:, :, 1::2, :]) / 2.0
            LH = (L[:, :, 0::2, :] - L[:, :, 1::2, :]) / 2.0
            HL = (H[:, :, 0::2, :] + H[:, :, 1::2, :]) / 2.0
            HH = (H[:, :, 0::2, :] - H[:, :, 1::2, :]) / 2.0
            return LL, LH, HL, HH

        loss = torch.tensor(0.0, device=pred.device)

        for level in range(2):
            _, LH_p, HL_p, HH_p = haar_dwt_2d(pred)
            _, LH_t, HL_t, HH_t = haar_dwt_2d(target)
            loss = loss + F.l1_loss(LH_p, LH_t) \
                        + F.l1_loss(HL_p, HL_t) \
                        + F.l1_loss(HH_p, HH_t)
            # 下一级对 LL 做
            pred = haar_dwt_2d(pred)[0]
            target = haar_dwt_2d(target)[0]

        return loss

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
    def estimate_noise_level(self, condition: torch.Tensor) -> torch.Tensor:
        """
        估计输入低剂量图像相对于干净图像的噪声水平。
        
        使用 condition 的局部方差作为噪声代理指标。
        高方差 → 高噪声 → 需要强力去噪 (高 CFG, 多 NFE)
        低方差 → 低噪声 → 需要保守去噪 (低 CFG, 少 NFE)
        
        Returns:
            noise_level: (B,) 归一化噪声水平 ∈ [0, 1]
        """
        B = condition.size(0)
        # 局部方差 (3×3) 作为噪声估计
        kernel = torch.ones(1, 1, 3, 3, device=condition.device) / 9.0
        local_mean = F.conv2d(condition, kernel, padding=1)
        local_sq_mean = F.conv2d(condition ** 2, kernel, padding=1)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
        
        # 全局归一化: 每张图的平均局部方差
        noise_level = local_var.mean(dim=[1, 2, 3])  # (B,)
        # 经验映射到 [0, 1]: var<1e-6 → 极干净, var>1e-3 → 极噪
        noise_level = (noise_level / 1e-3).clamp(0, 1)
        return noise_level

    @torch.no_grad()
    def generate_adaptive(self, condition: torch.Tensor, body_part: torch.Tensor,
                          base_steps: int = 50, base_cfg: float = 0.6) -> torch.Tensor:
        """
        CFG + NFE 自适应推理。
        
        根据输入噪声水平自动调整:
          - 干净输入 (高剂量) → 降低 CFG, 减少步数
          - 噪声输入 (低剂量) → 提高 CFG, 增加步数
        
        经验映射:
          noise=0.0 (1/2剂量) → cfg=0.2, nfe=20
          noise=0.5 (1/10剂量) → cfg=0.6, nfe=50
          noise=1.0 (1/100剂量) → cfg=1.0, nfe=75
        """
        noise_lvl = self.estimate_noise_level(condition)
        
        # 逐样本自适应参数
        adaptive_cfg = base_cfg * (0.3 + 0.7 * noise_lvl)  # [0.3*cfg, 1.0*cfg]
        adaptive_steps = torch.clamp((base_steps * (0.4 + 0.6 * noise_lvl)).long(), min=15, max=100)
        
        B = condition.size(0)
        outputs = []
        
        for i in range(B):
            cond_i = condition[i:i+1]
            bp_i = body_part[i:i+1]
            cfg_i = adaptive_cfg[i].item()
            steps_i = adaptive_steps[i].item()
            
            out_i = self.generate(cond_i, bp_i, steps=steps_i, cfg_scale=cfg_i)
            outputs.append(out_i)
        
        return torch.cat(outputs, dim=0)

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

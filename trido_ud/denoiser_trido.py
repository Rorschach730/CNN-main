"""
TriDo-JiT Denoiser: Flow Matching Wrapper with FGW Regularization
===================================================================
Wraps the TriDoJiT model with Flow Matching training and inference.
Uses v-prediction (velocity prediction) — same formulation as the _ud version.

Key extensions over the _ud Denoiser:
  - FGW (Fused Gromov-Wasserstein) loss as structural regularizer
  - Frequency-domain loss from GFP module
  - Sinogram consistency loss (optional)

Training: v_pred = (x_pred - z) / (1 - t), target = clean - condition
Inference: Heun's 2nd-order ODE solver with CFG (classifier-free guidance)

Reference: Flow Matching for Generative Modeling (Lipman et al., 2023)
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
    Flow Matching denoiser for TriDo-JiT.

    Training:
      - Primary: v-prediction Huber loss (Flow Matching)
      - Regularizer: FGW structural loss
      - Optional: frequency consistency loss, sino consistency loss

    Inference:
      - Heun's 2nd-order ODE solver
      - Classifier-Free Guidance (CFG)
      - Supports body-part conditioned sampling

    Args:
        args: Argument namespace containing:
            - img_size, patch_size, attn_dropout, proj_dropout
            - P_mean, P_std (logit-normal timestep sampling params)
            - cond_drop_prob (CFG dropout rate)
            - cfg_scale (inference guidance scale)
            - model_size ('Large', 'Base', 'Small')
            - fgw_weight, freq_weight (loss weights)
            - use_sino_domain, use_freq_domain
    """

    def __init__(self, args):
        super().__init__()

        self.patch_size = getattr(args, 'patch_size', 16)
        model_size = getattr(args, 'model_size', 'Base')
        model_key = f'TriDoJiT-{model_size}'

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
        self.P_mean = getattr(args, 'P_mean', -0.5)
        self.P_std = getattr(args, 'P_std', 1.2)
        self.cond_drop_prob = getattr(args, 'cond_drop_prob', 0.1)
        self.cfg_scale = getattr(args, 'cfg_scale', 2.0)

        # Loss weights
        self.fgw_weight = getattr(args, 'fgw_weight', 0.01)
        self.freq_weight = getattr(args, 'freq_weight', 0.005)
        self.sino_weight = getattr(args, 'sino_weight', 0.01)
        self.struct_weight = getattr(args, 'struct_weight', 0.01)

        # FGW loss (patch-based, efficient)
        self.fgw_loss = FGWLoss(
            patch_size=16, stride=8, alpha=0.5,
            reg=0.1, feature_weight=1.0
        )

        # Structural consistency (fast Gram-based alternative)
        self.struct_loss = StructuralConsistencyLoss(weight=1.0)

        # EMA
        accum_iter = getattr(args, 'accum_iter', 1)
        self.ema_decay = 0.999 ** accum_iter
        self.ema_params = [p.clone().detach() for p in self.net.parameters()]

    def sample_t(self, n, device):
        """Logit-normal timestep sampling."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, target, condition, body_part):
        """
        Training forward pass with Flow Matching + FGW regularization.

        Args:
            target: (B, 1, H, W) — clean full-dose PET image
            condition: (B, 1, H, W) — low-dose PET image
            body_part: (B,) LongTensor — body part category (0=brain, 1=chest, 2=abdomen)

        Returns:
            total_loss: scalar (v-prediction Huber + FGW + freq + struct)
        """
        bsz = target.size(0)
        device = target.device

        # --- Flow Matching: sample t, interpolate ---
        t = self.sample_t(bsz, device=device)
        t_reshape = t.view(-1, 1, 1, 1)

        # Interpolation: z = t * target + (1-t) * condition
        z = t_reshape * target + (1 - t_reshape) * condition

        # CFG: randomly drop condition
        drop_mask = (torch.rand(bsz, 1, 1, 1, device=device) > self.cond_drop_prob).float()
        condition_input = condition * drop_mask

        # Model input: concatenate z and condition
        model_input = torch.cat([z, condition_input], dim=1)  # (B, 2, H, W)

        # --- Forward through TriDoJiT ---
        x_pred = self.net(model_input, t.flatten(), body_part)  # (B, 1, H, W)

        # --- v-prediction loss ---
        denom = torch.clamp(1.0 - t_reshape, min=0.05)
        v_pred = (x_pred - z) / denom
        v_target = target - condition

        loss_v = F.smooth_l1_loss(v_pred, v_target, beta=0.1)

        # --- FGW structural loss (on predicted clean image) ---
        # Reconstruct clean prediction from v_pred
        with torch.no_grad():
            x_clean_from_v = z + v_pred * (1.0 - t_reshape)

        loss_fgw = self.fgw_loss(x_pred, target)

        # --- Structural consistency (Gram-based, faster) ---
        loss_struct = self.struct_loss(x_pred, target)

        # --- Frequency loss (from GFP) ---
        if hasattr(self.net, 'gfp') and self.net.gfp is not None:
            loss_freq = self.net.gfp.compute_frequency_loss(x_pred, target)
        else:
            loss_freq = torch.tensor(0.0, device=device)

        # --- Sinogram consistency loss ---
        loss_sino = torch.tensor(0.0, device=device)
        if hasattr(self.net, 'sino_bridge') and self.net.sino_bridge is not None:
            # Forward project both and compare in sinogram domain
            with torch.no_grad():
                sino_target = self.net.sino_bridge.forward_project(target)
            sino_pred = self.net.sino_bridge.forward_project(x_pred)
            loss_sino = F.l1_loss(sino_pred, sino_target)

        # --- Total loss ---
        total_loss = (
            loss_v
            + self.fgw_weight * loss_fgw
            + self.freq_weight * loss_freq
            + self.struct_weight * loss_struct
            + self.sino_weight * loss_sino
        )

        # Store for logging
        self._last_losses = {
            'v_loss': loss_v.item(),
            'fgw_loss': loss_fgw.item(),
            'freq_loss': loss_freq.item(),
            'struct_loss': loss_struct.item(),
            'sino_loss': loss_sino.item(),
            'total': total_loss.item(),
        }

        return total_loss

    def get_last_losses(self):
        """Return the breakdown of the last computed loss."""
        return getattr(self, '_last_losses', {})

    @torch.no_grad()
    def update_ema(self):
        current_device = next(self.net.parameters()).device
        if self.ema_params[0].device != current_device:
            self.ema_params = [p.to(current_device) for p in self.ema_params]
        for targ, src in zip(self.ema_params, self.net.parameters()):
            targ.mul_(self.ema_decay).add_(src, alpha=1 - self.ema_decay)

    def get_ema_state_dict(self):
        ema_dict = {}
        for (name, _), ema_param in zip(self.net.named_parameters(), self.ema_params):
            ema_dict[name] = ema_param
        return ema_dict

    def load_ema_state_dict(self, ema_dict):
        for (name, param), ema_param in zip(self.net.named_parameters(), self.ema_params):
            if name in ema_dict:
                ema_param.data.copy_(ema_dict[name].data)

    @torch.no_grad()
    def generate(self, condition, body_part, steps=50, cfg_scale=None):
        """
        Generate denoised PET image via Flow Matching ODE sampling.

        Uses Heun's 2nd-order method with Classifier-Free Guidance.

        Args:
            condition: (B, 1, H, W) — low-dose PET image
            body_part: (B,) LongTensor — body part category (0=brain, 1=chest, 2=abdomen)
            steps: Number of ODE integration steps
            cfg_scale: CFG scale (defaults to self.cfg_scale)

        Returns:
            denoised: (B, 1, H, W) — high-quality PET image
        """
        device = condition.device
        bsz = condition.size(0)

        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        z = condition.clone()
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        uncond_condition = torch.zeros_like(condition)

        # [双路并行适配]：将 body_part 复制一份（CFG 条件+无条件）
        body_in = torch.cat([body_part, body_part], dim=0)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            t_curr_batch = t_curr.repeat(bsz)

            z_in = torch.cat([z, z], dim=0)
            cond_in = torch.cat([condition, uncond_condition], dim=0)
            t_in = torch.cat([t_curr_batch, t_curr_batch], dim=0)

            x_pred_curr = self.net(
                torch.cat([z_in, cond_in], dim=1), t_in, body_in
            )
            denom_curr = torch.clamp(1.0 - t_curr, min=0.05)
            v_curr_all = (x_pred_curr - z_in) / denom_curr

            v_curr_cond, v_curr_uncond = v_curr_all.chunk(2, dim=0)
            v_curr = v_curr_uncond + cfg_scale * (v_curr_cond - v_curr_uncond)

            z_tmp = z + v_curr * dt

            if i == steps - 1:
                z = z_tmp
            else:
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

                z = z + 0.5 * (v_curr + v_next) * dt

        return z

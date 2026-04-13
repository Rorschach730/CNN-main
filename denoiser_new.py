import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()

        # [拓展实装 1]: 下调 Patch Size (默认 8)，提升高频纹理感知野
        self.patch_size = getattr(args, 'patch_size', 8)

        # [物理降维]: in_channels 退行至 2 (Z: 1通道 + Condition: 1通道)
        self.net = JiT_models['JiT-Large'](
            input_size=args.img_size,
            patch_size=self.patch_size,
            in_channels=2,
            out_channels=1,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )

        self.img_size = args.img_size

        # [方案三实装]: 默认中心向低噪区偏移，压制结构重建
        self.P_mean = getattr(args, 'P_mean', -0.5)
        self.P_std = getattr(args, 'P_std', 1.2)

        # [CFG 训练端]: 10% 概率触发物理致盲 (Condition Dropout)
        self.cond_drop_prob = getattr(args, 'cond_drop_prob', 0.1)

        accum_iter = getattr(args, 'accum_iter', 1)
        self.ema_decay = 0.999 ** accum_iter
        self.ema_params = [p.clone().detach() for p in self.net.parameters()]

    def sample_t(self, n, device):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, target, condition):
        bsz = target.size(0)
        device = target.device

        t = self.sample_t(bsz, device=device)
        t_reshape = t.view(-1, 1, 1, 1)

        cond_center = condition
        z = t_reshape * target + (1 - t_reshape) * cond_center

        # [CFG 核心逻辑]: 随机生成掩码，按概率抹除 condition 信号
        drop_mask = (torch.rand(bsz, 1, 1, 1, device=device) > self.cond_drop_prob).float()
        condition_input = cond_center * drop_mask

        model_input = torch.cat([z, condition_input], dim=1)

        # JiT 原生预测
        x_pred = self.net(model_input, t.flatten())

        denom = torch.clamp(1.0 - t_reshape, min=0.05)
        v_pred = (x_pred - z) / denom
        v_target = target - cond_center

        # [终极物理约束]: 剔除所有回归惩罚，仅保留纯正流形 MSE
        loss_mse = F.mse_loss(v_pred, v_target)

        return loss_mse

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
        # 仅恢复 EMA 缓冲池，严禁触碰真实网络防动量错位
        for (name, param), ema_param in zip(self.net.named_parameters(), self.ema_params):
            if name in ema_dict:
                ema_param.data.copy_(ema_dict[name].data)

    @torch.no_grad()
    def generate(self, condition, steps=50, cfg_scale=2.0):
        # [CFG 推理端]: 启用无条件流形与条件流形的双路并行计算
        device = condition.device
        bsz = condition.size(0)

        # 允许通过对象属性或参数动态传入 cfg_scale
        cfg_scale = getattr(self, 'cfg_scale', cfg_scale)

        z = condition.clone()
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

        # 预先分配无条件锚点
        uncond_condition = torch.zeros_like(condition)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            t_curr_batch = t_curr.repeat(bsz)

            # --- Heun Step 1: Euler 预测 (双路并行) ---
            z_in = torch.cat([z, z], dim=0)
            cond_in = torch.cat([condition, uncond_condition], dim=0)
            t_in = torch.cat([t_curr_batch, t_curr_batch], dim=0)

            x_pred_curr = self.net(torch.cat([z_in, cond_in], dim=1), t_in)
            denom_curr = torch.clamp(1.0 - t_curr, min=0.05)
            v_curr_all = (x_pred_curr - z_in) / denom_curr

            # 张量切割与外推
            v_curr_cond, v_curr_uncond = v_curr_all.chunk(2, dim=0)
            v_curr = v_curr_uncond + cfg_scale * (v_curr_cond - v_curr_uncond)

            z_tmp = z + v_curr * dt

            if i == steps - 1:
                z = z_tmp
            else:
                # --- Heun Step 2: 梯形校正 (双路并行) ---
                t_next_batch = t_next.repeat(bsz)
                z_tmp_in = torch.cat([z_tmp, z_tmp], dim=0)
                t_next_in = torch.cat([t_next_batch, t_next_batch], dim=0)

                x_pred_next = self.net(torch.cat([z_tmp_in, cond_in], dim=1), t_next_in)
                denom_next = torch.clamp(1.0 - t_next, min=0.05)
                v_next_all = (x_pred_next - z_tmp_in) / denom_next

                # 张量切割与外推
                v_next_cond, v_next_uncond = v_next_all.chunk(2, dim=0)
                v_next = v_next_uncond + cfg_scale * (v_next_cond - v_next_uncond)

                z = z + 0.5 * (v_curr + v_next) * dt

        return z
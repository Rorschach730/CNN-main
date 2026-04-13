import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_jit import JiT_models


# ==========================================
#        Differentiable SSIM Engine
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_ssim(img1, img2, window, val_range=1.0):
    # [物理修复 1]: 默认 val_range 被收束为严格的 1.0，适配临床归一化张量
    channel = img1.size(1)
    window_size = window.size(2)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.relu(F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq)
    sigma2_sq = F.relu(F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()

        # [物理降维]: in_channels 强行退行至 2 (Z: 1通道 + Condition: 1通道)
        self.net = JiT_models['JiT-Large'](
            input_size=args.img_size,
            in_channels=2,
            out_channels=1,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )

        self.img_size = args.img_size
        self.P_mean = args.P_mean
        self.P_std = args.P_std

        # [物理修复 3]: 探测全局 accum_iter 并进行 EMA 动量时钟补偿
        accum_iter = getattr(args, 'accum_iter', 1)
        self.ema_decay = 0.999 ** accum_iter

        self.ema_params = [p.clone().detach() for p in self.net.parameters()]
        self.register_buffer("ssim_window", create_window(11, 1))

    def sample_t(self, n, device):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, target, condition):
        t = self.sample_t(target.size(0), device=target.device)
        t_reshape = t.view(-1, 1, 1, 1)

        # [物理降维]: 纯 2D 架构，condition 直接作为中心层，无需再切片
        cond_center = condition
        z = t_reshape * target + (1 - t_reshape) * cond_center

        model_input = torch.cat([z, condition], dim=1)

        # [JiT理论映射]: 网络直接输出干净图像 x_pred
        x_pred = self.net(model_input, t.flatten())

        # 将 x_pred 重参数化回 v_pred，计算损失，分母防止除零
        denom = torch.clamp(1.0 - t_reshape, min=0.05)
        v_pred = (x_pred - z) / denom
        v_target = target - cond_center

        # 1. 振幅标量约束 (MSE)
        loss_mse = F.mse_loss(v_pred, v_target)

        # 2. 高维摆线定向约束 (Cosine) + [物理修复 2: 物理真空掩码防幻觉]
        v_pred_flat = v_pred.view(v_pred.shape[0], -1)
        v_target_flat = v_target.view(v_target.shape[0], -1)

        v_target_norm = torch.norm(v_target_flat, p=2, dim=1)
        cos_sim = F.cosine_similarity(v_pred_flat, v_target_flat, dim=1)

        valid_mask = (v_target_norm > 1e-5).float()
        loss_dir_raw = (1.0 - cos_sim) * valid_mask
        loss_dir = loss_dir_raw.sum() / (valid_mask.sum() + 1e-8)

        # 3. 正交物理平衡
        loss_v = loss_mse + 0.05 * loss_dir

        loss_l1 = F.l1_loss(x_pred, target)
        loss_ssim = 1.0 - compute_ssim(x_pred, target, self.ssim_window, val_range=1.0)

        loss_total = loss_v + loss_l1 + 0.5 * loss_ssim
        return loss_total

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
        # [断点续训修复]: 仅恢复 EMA 影子缓冲池，严禁触碰真实的 self.net，以防动量错位
        for (name, param), ema_param in zip(self.net.named_parameters(), self.ema_params):
            if name in ema_dict:
                # 同步更新缓冲池
                ema_param.data.copy_(ema_dict[name].data)

    @torch.no_grad()
    def generate(self, condition, steps=50):
        device = condition.device
        bsz = condition.size(0)

        # [物理降维]: condition 本身即为推演起点，直接克隆
        z = condition.clone()
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            # ---------------------------------------------------------
            # Heun 二阶最优传输求解器 (2nd-Order Runge-Kutta)
            # ---------------------------------------------------------
            # Step 1: Euler 预测 (Euler Predictor)
            x_pred_curr = self.net(torch.cat([z, condition], dim=1), t_curr.repeat(bsz))
            denom_curr = torch.clamp(1.0 - t_curr, min=0.05)
            v_curr = (x_pred_curr - z) / denom_curr
            z_tmp = z + v_curr * dt

            if i == steps - 1:
                # 抵达物理流形终点，最后一步退化为一阶 Euler 以防止越界
                z = z_tmp
            else:
                # Step 2: 梯形校正 (Trapezoidal Corrector)
                x_pred_next = self.net(torch.cat([z_tmp, condition], dim=1), t_next.repeat(bsz))
                denom_next = torch.clamp(1.0 - t_next, min=0.05)
                v_next = (x_pred_next - z_tmp) / denom_next
                z = z + 0.5 * (v_curr + v_next) * dt

        return z
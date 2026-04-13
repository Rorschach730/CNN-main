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


def compute_ssim(img1, img2, window, val_range=2.0):
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

        self.ema_decay = 0.999
        self.ema_params = [p.clone().detach() for p in self.net.parameters()]
        self.register_buffer("ssim_window", create_window(11, 1))

    def sample_t(self, n, device):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, target, condition):
        t = self.sample_t(target.size(0), device=target.device)
        t_reshape = t.view(-1, 1, 1, 1)

        # I2I Optimal Transport Flow
        z = t_reshape * target + (1 - t_reshape) * condition

        model_input = torch.cat([z, condition], dim=1)

        v_pred = self.net(model_input, t.flatten())
        v_target = target - condition

        loss_v = F.mse_loss(v_pred, v_target)

        x_pred = z + (1 - t_reshape) * v_pred

        loss_l1 = F.l1_loss(x_pred, target)
        ssim_val = compute_ssim(x_pred, target, self.ssim_window, val_range=2.0)
        loss_ssim = 1.0 - ssim_val

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
        for (name, _), ema_param in zip(self.net.named_parameters(), self.ema_params):
            if name in ema_dict:
                ema_param.data.copy_(ema_dict[name].data)

    @torch.no_grad()
    def generate(self, condition, steps=5): # [修复] 强制下降到 5 步，拒绝过度平滑
        device = condition.device
        bsz = condition.size(0)

        z = condition.clone()
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            # [修复] 极简一阶 Euler 积分，保真高频特征
            v_curr = self.net(torch.cat([z, condition], dim=1), t_curr.repeat(bsz))
            dt = t_next - t_curr
            z = z + v_curr * dt

        return z
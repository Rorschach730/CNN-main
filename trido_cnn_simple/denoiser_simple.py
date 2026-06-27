"""
SimpleDenoiser: 简化训练/推理包装器
=====================================
TriDoSimpleCNN 的训练和推理接口。无扩散、无 ODE、无 CFG。

训练: output = model(condition), loss = L1(output, target)
推理: output = model(condition) 一次前馈
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from trido_cnn_simple.model_simple import TriDoSimple_models
except ImportError:
    from model_simple import TriDoSimple_models


class SimpleDenoiser(nn.Module):
    """
    简化去噪器（无扩散）。

    训练前向:
        x_pred = net(condition)
        loss = L1(x_pred, target)

    推理:
        result = net(condition)

    Args:
        args: Argument namespace
    """

    def __init__(self, args):
        super().__init__()

        model_size = getattr(args, 'model_size', 'Base')
        model_key = f'TriDoSimple-{model_size}'

        self.net = TriDoSimple_models[model_key](
            input_size=args.img_size,
            n_views=getattr(args, 'n_views', 96),
            use_sino_domain=getattr(args, 'use_sino_domain', True),
            use_freq_domain=getattr(args, 'use_freq_domain', True),
        )

        self.img_size = args.img_size

    def forward(self, condition: torch.Tensor,
                target: torch.Tensor = None) -> torch.Tensor:
        """
        训练前向：condition → 去噪 → L1 loss。

        Args:
            condition: (B, 1, H, W) 低剂量 PET
            target:    (B, 1, H, W) 全剂量 PET (训练时提供)

        Returns:
            loss 标量 (训练) 或 output (推理)
        """
        x_pred = self.net(condition)

        if target is not None:
            loss = F.smooth_l1_loss(x_pred, target, beta=0.1)
            return loss
        else:
            return x_pred

    @torch.no_grad()
    def generate(self, condition: torch.Tensor) -> torch.Tensor:
        """推理：一次前馈去噪"""
        self.net.eval()
        return self.net(condition)

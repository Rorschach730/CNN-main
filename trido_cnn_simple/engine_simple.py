"""
Simple Training Engine
======================
极简训练循环：前馈 → L1 loss → 反向传播。无 AMP、无梯度累加（batch 大了不需要）。
"""

import math
import sys
import torch


def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    """
    训练一个 epoch。

    Args:
        model:      SimpleDenoiser
        data_loader: DataLoader → (target, condition, body_part)
        optimizer:  AdamW
        device:     torch device
        epoch:      当前 epoch
        args:       参数命名空间

    Returns:
        stats: {'loss': avg_loss}
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (targets, conditions, _body_parts) in enumerate(data_loader):
        targets = targets.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)

        loss = model(conditions, targets)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch:03d} [{batch_idx:04d}/{len(data_loader)}] "
                  f"loss: {loss.item():.6f}")

    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch:03d} avg_loss: {avg_loss:.6f}")
    return {'loss': avg_loss}

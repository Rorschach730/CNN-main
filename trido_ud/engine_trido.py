"""
TriDo-CNN Training Engine
===========================
Extended training loop with multi-loss logging and three-domain monitoring.

Adapted from engine_jit_ud.py with additions:
  - Per-loss-component logging (v_loss, fgw_loss, freq_loss, struct_loss, sino_loss)
  - Gradient accumulation with safety mechanisms
  - AMP (Automatic Mixed Precision) with bfloat16
  - Domain-specific monitoring hooks
"""

import math
import sys
import torch
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    """
    Train TriDo-CNN for one epoch.

    Unpacks (target, condition, body_part) tuples, runs three-domain forward pass,
    and accumulates multi-component loss (Flow Matching + FGW + frequency + structure).

    Args:
        model: TriDoDenoiser instance
        data_loader: PyTorch DataLoader yielding (target, condition, body_part)
        optimizer: AdamW optimizer
        device: torch device
        epoch: Current epoch number
        args: Argument namespace

    Returns:
        stats: dict with averaged losses
    """
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = getattr(args, 'accum_iter', 4)
    optimizer.zero_grad()

    # Track per-component losses
    loss_components = {}

    for data_iter_step, (targets, conditions, body_parts) in enumerate(
        metric_logger.log_every(data_loader, 20, header)
    ):
        targets = targets.to(device, non_blocking=True).to(torch.float32)
        conditions = conditions.to(device, non_blocking=True).to(torch.float32)
        body_parts = body_parts.to(device, non_blocking=True).to(torch.long)

        # [AMP 加速实装]: bfloat16 混合精度
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model(targets, conditions, body_parts)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Collect loss breakdown for logging
        component_losses = model.get_last_losses()
        for k, v in component_losses.items():
            if k not in loss_components:
                loss_components[k] = []
            loss_components[k].append(v)

        # [核心防爆 1: 梯度累加]
        loss = loss / accum_iter
        loss.backward()

        if ((data_iter_step + 1) % accum_iter == 0) or ((data_iter_step + 1) == len(data_loader)):
            # [核心防爆 3: 时钟对齐] LR 调度
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

            # [核心防爆 2: 梯度范数裁剪]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            model.update_ema()

        metric_logger.update(loss=loss_value)

    # Print epoch summary
    print("Averaged stats:", metric_logger)

    # Build stats dict with all components
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Add per-component averages
    if loss_components:
        print("\n--- Loss Component Breakdown ---")
        for k, vals in loss_components.items():
            avg_val = sum(vals) / len(vals)
            print(f"  {k}: {avg_val:.6f}")
            stats[k] = avg_val

    return stats

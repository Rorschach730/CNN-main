import math
import sys
import torch
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    # 提取累加步数，默认 4
    accum_iter = getattr(args, 'accum_iter', 4)
    optimizer.zero_grad()

    # [架构升级] 解包三元组，接驳剂量标量
    for data_iter_step, (targets, conditions, doses) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        targets = targets.to(device, non_blocking=True).to(torch.float32)
        conditions = conditions.to(device, non_blocking=True).to(torch.float32)
        doses = doses.to(device, non_blocking=True).to(torch.float32)

        # [AMP 加速实装]: 启用 bfloat16 混合精度计算，调用 Tensor Core
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # 将剂量标量透传至模型
            loss = model(targets, conditions, doses)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # [核心防爆 1: 梯度累加] 损失归一化以保证数学等价
        loss = loss / accum_iter
        loss.backward()

        # 仅在满足累加步数或到达 Epoch 终点时，执行真实的物理更新
        if ((data_iter_step + 1) % accum_iter == 0) or ((data_iter_step + 1) == len(data_loader)):
            # [核心防爆 3: 时钟对齐] LR 调度被强制同步到真实的 optimizer.step() 频率
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            # [核心防爆 2: 梯度范数裁剪] 将极端流形的离群梯度按向量等比例缩小，保住更新方向
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            model.update_ema()

        metric_logger.update(loss=loss_value)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
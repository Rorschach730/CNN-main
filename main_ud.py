import argparse
import os
import time
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

from util.pet_dataset_ud import PETDenoisingDataset
from denoiser_ud import Denoiser
from engine_jit_ud import train_one_epoch
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('JiT PET Denoising', add_help=False)
    # [极限算力压榨实装]: Batch 24 + Accum 11，等效总批次 264
    parser.add_argument('--batch_size', default=24, type=int, help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=11, type=int, help='Gradient Accumulation steps')
    # [生命周期截断]: 强行设定为 200 轮
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # [参数修正]: 强制对齐输入尺寸至 256
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)

    parser.add_argument('--P_mean', default=-0.5, type=float)
    parser.add_argument('--P_std', default=1.2, type=float)

    # [显存防爆与架构对齐]: Patch Size 锁定 16
    parser.add_argument('--patch_size', default=16, type=int, help='Patch tokenization size')
    parser.add_argument('--cond_drop_prob', default=0.1, type=float, help='CFG 空间条件致盲率')
    parser.add_argument('--cfg_scale', default=2.0, type=float, help='推理阶段的无分类器引导外推强度')

    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=0.001, type=float)

    # [数据源修正]: 指向物理清洗落盘目录
    # (注意：如果 cleaner 落盘在 ../processed_data_udpet，请在命令行中加上 --data_path ../processed_data_udpet)
    parser.add_argument('--data_path', default='./processed_data_udpet', type=str)
    parser.add_argument('--output_dir', default='./output_dir_ud', help='模型保存路径')
    parser.add_argument('--device', default='cuda', help='Device: GPU')
    parser.add_argument('--resume', default='', help='Resume checkpoint path')
    parser.add_argument('--warmup_epochs', type=int, default=10)

    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    return parser


def main(args):
    print(f"Start training on {args.device} with batch size {args.batch_size}, accum_iter {args.accum_iter}")
    device = torch.device(args.device)

    torch.set_float32_matmul_precision('high')

    # 初始化多剂量数据集
    dataset_train = PETDenoisingDataset(os.path.join(args.data_path, 'train'), img_size=args.img_size)

    # [回归纯粹]: 移除 Chunked Sampler，直接启动 PyTorch 原生 DataLoader 的全局 shuffle
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,  # 开启全局随机打乱
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    model = Denoiser(args)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    train_losses = []
    log_file_path = os.path.join(args.output_dir, 'training_log.txt')

    # [断点续训逻辑]
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        if 'model_ema' in ckpt:
            model.load_ema_state_dict(ckpt['model_ema'])

        start_epoch = ckpt.get('epoch', -1) + 1
        print(f"Resumed from {args.resume} (Starting at Epoch {start_epoch})")

        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                for line in f:
                    if "Train Loss:" in line:
                        try:
                            loss_val = float(line.strip().split('Train Loss:')[1])
                            train_losses.append(loss_val)
                        except:
                            pass
        train_losses = train_losses[:start_epoch]

        with open(log_file_path, 'a') as f:
            f.write(f"\n--- Resumed Training on {args.device} from Epoch {start_epoch} ---\n")
    else:
        if os.path.exists(log_file_path):
            open(log_file_path, 'w').close()
        with open(log_file_path, 'a') as f:
            f.write(f"\n--- Start Fresh Training on {args.device} ---\n")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # 将三元组 (target, condition, dose) 的提取逻辑交由 engine_jit 处理
        train_stats = train_one_epoch(model, dataloader_train, optimizer, device, epoch, args)

        ep_loss = train_stats['loss']
        train_losses.append(ep_loss)

        with open(log_file_path, 'a') as f:
            f.write(f"Epoch {epoch:03d} | Train Loss: {ep_loss:.6f}\n")

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, label='v-prediction Huber Loss', color='blue',
                 linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value (Log Scale)')
        plt.title('Training Loss Convergence Curve')
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'), dpi=150)
        plt.close()

        # [分段存储机制实装]
        is_save_epoch = False
        if epoch < 100:
            if epoch % 50 == 0:
                is_save_epoch = True
        else:
            if epoch % 10 == 0:
                is_save_epoch = True

        # 强制保存最后一个 epoch 防止越界丢失
        if epoch == args.epochs - 1:
            is_save_epoch = True

        if is_save_epoch:
            save_path = os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
            torch.save({
                'model': model.net.state_dict(),
                'model_ema': model.get_ema_state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }, save_path)
            print(f"Saved checkpoint to {save_path} (including EMA)")

    total_time = time.time() - start_time
    print(f'Training time: {total_time / 3600:.2f} hours')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
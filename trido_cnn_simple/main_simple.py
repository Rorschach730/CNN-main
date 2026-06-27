"""
TriDo-Simple Training Script
==============================
极简三域 CNN 去噪训练入口。无扩散、无 ODE、无辅助损失。

用法:
    python trido_cnn_simple/main_simple.py --data_path E:/processed_data_trido --output_dir ./simple_output

与 TriDo-CNN (扩散版) 的区别:
    - 无 timestep 采样、无 ODE、无 CFG
    - 训练: 纯 L1 loss, 无 FGW/HALO/GFP/sino 辅助损失
    - 推理: model(condition) 一次前馈
"""

import argparse
import os
import time
import torch
import numpy as np
from pathlib import Path
import random

try:
    from trido_cnn_simple.denoiser_simple import SimpleDenoiser
    from trido_cnn_simple.engine_simple import train_one_epoch
    from trido_ud.pet_dataset_trido import TriDoPETDataset
except ImportError:
    from denoiser_simple import SimpleDenoiser
    from engine_simple import train_one_epoch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from trido_ud.pet_dataset_trido import TriDoPETDataset


def get_args_parser():
    parser = argparse.ArgumentParser('TriDo-Simple CNN Denoising', add_help=True)

    # --- Training ---
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # --- Architecture ---
    parser.add_argument('--model_size', default='Base', type=str,
                        choices=['Base', 'Small', 'Tiny'])
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--n_views', default=96, type=int,
                        help='Radon projection views')

    # --- Domain control ---
    parser.add_argument('--use_sino_domain', action='store_true', default=True)
    parser.add_argument('--no_sino_domain', action='store_false', dest='use_sino_domain')
    parser.add_argument('--use_freq_domain', action='store_true', default=True)
    parser.add_argument('--no_freq_domain', action='store_false', dest='use_freq_domain')

    # --- Data ---
    parser.add_argument('--data_path', default='E:/processed_data_trido/', type=str)
    parser.add_argument('--output_dir', default='./simple_output', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--prefetch_factor', default=6, type=int)
    parser.add_argument('--virtual_epoch_ratio', default=0.05, type=float)

    # --- Misc ---
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    print("=" * 60)
    print("TriDo-Simple: 三域简化解耦 CNN 去噪（无扩散）")
    print("=" * 60)
    print(f"  Device:           {args.device}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Virtual Ratio:    {args.virtual_epoch_ratio * 100:.1f}%")
    print(f"  Model size:       {args.model_size}")
    print(f"  Sino domain:      {args.use_sino_domain}")
    print(f"  Freq domain:      {args.use_freq_domain}")
    print(f"  n_views:          {args.n_views}")
    print(f"  Data path:        {args.data_path}")
    print(f"  Output dir:       {args.output_dir}")
    print("=" * 60)

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')

    # --- Dataset ---
    dataset_train = TriDoPETDataset(
        os.path.join(args.data_path, 'train'),
        img_size=args.img_size,
        virtual_epoch_ratio=args.virtual_epoch_ratio,
        seed=args.seed
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor
    )

    # --- Model ---
    model = SimpleDenoiser(args)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {trainable_params / 1e6:.2f}M")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Resume ---
    start_epoch = 0
    train_losses = []
    log_file_path = os.path.join(args.output_dir, 'training_log.txt')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', -1) + 1
    else:
        with open(log_file_path, 'w') as f:
            f.write("--- TriDo-Simple Fresh Training ---\n")

    # --- Training loop ---
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        dataset_train.set_epoch(epoch)
        stats = train_one_epoch(model, dataloader_train, optimizer, device, epoch, args)

        ep_loss = stats['loss']
        train_losses.append(ep_loss)

        # Log
        with open(log_file_path, 'a') as f:
            f.write(f"Epoch {epoch:03d} | Loss: {ep_loss:.6f}\n")
        print(f"Epoch {epoch:03d} | Loss: {ep_loss:.6f}")

        # Checkpoint
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
            torch.save({
                'model': model.net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }, save_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time / 3600:.2f} hours")

    # Final checkpoint
    final_path = os.path.join(args.output_dir, 'checkpoint-final.pth')
    torch.save({
        'model': model.net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.epochs - 1,
        'args': args
    }, final_path)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)

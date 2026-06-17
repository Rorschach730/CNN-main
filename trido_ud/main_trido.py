"""
TriDo-CNN Training Script
===========================
Main entry point for training the Triple-Domain CNN
for low-dose PET denoising.

Usage:
    # Full triple-domain training
    python trido_ud/main_trido.py --data_path ./processed_data_trido --output_dir ./trido_output

    # Image-only (ablation: no sino domain)
    python trido_ud/main_trido.py --no_sino_domain

    # Image + GFP (ablation: no sino domain, keep GFP)
    python trido_ud/main_trido.py --no_sino_domain --use_freq_domain

    # Pure JiT baseline (disable both sino and freq domains)
    python trido_ud/main_trido.py --no_sino_domain --no_freq_domain

Key features:
    - Three-domain architecture (sinogram + image + frequency)
    - Flow Matching v-prediction with FGW regularization
    - Body part embedding for anatomy-adaptive denoising
    - AMP (bfloat16) training with gradient accumulation
    - EMA weight tracking
    - Multi-component loss logging
"""

import argparse
import os
import time
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

try:
    from trido_ud.pet_dataset_trido import TriDoPETDataset
    from trido_ud.denoiser_trido import TriDoDenoiser
    from trido_ud.engine_trido import train_one_epoch
except ImportError:
    from pet_dataset_trido import TriDoPETDataset
    from denoiser_trido import TriDoDenoiser
    from engine_trido import train_one_epoch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('TriDo-CNN PET Denoising', add_help=True)

    # --- Training ---
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=32, type=int, help='Gradient accumulation steps')
    parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='LR warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR')
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'constant'])

    # --- Architecture ---
    parser.add_argument('--model_size', default='Large', type=str,
                        choices=['Large', 'Base', 'Small'],
                        help='TriDo-CNN model size')
    parser.add_argument('--img_size', default=256, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch tokenization size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout')

    # --- Domain control ---
    parser.add_argument('--use_sino_domain', action='store_true', default=True,
                        help='Enable sinogram domain processing')
    parser.add_argument('--no_sino_domain', action='store_false', dest='use_sino_domain',
                        help='Disable sinogram domain (ablation)')
    parser.add_argument('--use_freq_domain', action='store_true', default=True,
                        help='Enable frequency domain (GFP)')
    parser.add_argument('--no_freq_domain', action='store_false', dest='use_freq_domain',
                        help='Disable frequency domain (ablation)')

    # --- Flow Matching ---
    parser.add_argument('--P_mean', default=-0.5, type=float, help='Logit-normal mean')
    parser.add_argument('--P_std', default=1.2, type=float, help='Logit-normal std')
    parser.add_argument('--cond_drop_prob', default=0.1, type=float, help='CFG condition dropout')
    parser.add_argument('--cfg_scale', default=2.0, type=float, help='CFG guidance scale')

    # --- Loss weights ---
    parser.add_argument('--fgw_weight', default=0.01, type=float, help='FGW structural loss weight')
    parser.add_argument('--freq_weight', default=0.005, type=float, help='Frequency loss weight')
    parser.add_argument('--struct_weight', default=0.01, type=float, help='Structural consistency loss weight')
    parser.add_argument('--sino_weight', default=0.01, type=float, help='Sinogram consistency loss weight')

    # --- Data ---
    parser.add_argument('--data_path', default='I:/processed_data_trido/', type=str,
                        help='Path to processed PET data')
    parser.add_argument('--output_dir', default='./trido_output', type=str,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--device', default='cuda', help='Device (cuda/mps/cpu)')
    parser.add_argument('--resume', default='', help='Resume checkpoint path')
    parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')
    parser.add_argument('--prefetch_factor', default=4, type=int, help='DataLoader prefetch')

    # --- Misc ---
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--noise_scale', default=1.0, type=float, help='Noise scale factor')
    
    # --- Virtual Epoch ---
    parser.add_argument('--virtual_epoch_ratio', default=0.10, type=float, help='Fraction of dataset to use per virtual epoch (e.g., 0.10 = 10%)')

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    print("=" * 70)
    print("TriDo-CNN: Triple-Domain CNN for PET Denoising")
    print("=" * 70)
    print(f"  Device:           {args.device}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Accumulation:     {args.accum_iter}")
    print(f"  Effective batch:  {args.batch_size * args.accum_iter}")
    print(f"  Epochs:           {args.epochs} (Virtual)")
    print(f"  Virtual Ratio:    {args.virtual_epoch_ratio * 100:.1f}%")
    print(f"  Model size:       {args.model_size}")
    print(f"  Sino domain:      {args.use_sino_domain}")
    print(f"  Freq domain:      {args.use_freq_domain}")
    print(f"  FGW weight:       {args.fgw_weight}")
    print(f"  Data path:        {args.data_path}")
    print(f"  Output dir:       {args.output_dir}")
    print("=" * 70)

    set_seed(args.seed)
    device = torch.device(args.device)

    # Set precision
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
    model = TriDoDenoiser(args)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {trainable_params / 1e6:.2f}M")
    print(f"  - Sinogram Encoder:  {sum(p.numel() for p in model.net.sino_encoder.parameters()) / 1e6:.2f}M" if args.use_sino_domain else "")
    print(f"  - GFP Module:        {sum(p.numel() for p in model.net.gfp.parameters()) / 1e6:.2f}M" if args.use_freq_domain else "")
    print()

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Resume ---
    start_epoch = 0
    train_losses = []
    log_file_path = os.path.join(args.output_dir, 'training_log.txt')

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        if 'model_ema' in ckpt:
            model.load_ema_state_dict(ckpt['model_ema'])

        start_epoch = ckpt.get('epoch', -1) + 1
        print(f"  Starting at Epoch {start_epoch}")

        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                for line in f:
                    if "Total Loss:" in line:
                        try:
                            loss_val = float(line.strip().split('Total Loss:')[1].split(',')[0])
                            train_losses.append(loss_val)
                        except:
                            pass
        train_losses = train_losses[:start_epoch]

        with open(log_file_path, 'a') as f:
            f.write(f"\n--- Resumed Training from Epoch {start_epoch} ---\n")
    else:
        # Fresh start
        if os.path.exists(log_file_path):
            open(log_file_path, 'w').close()
        with open(log_file_path, 'a') as f:
            f.write("--- TriDo-CNN Fresh Training ---\n")
            f.write(f"Model: {args.model_size}, Sino={args.use_sino_domain}, "
                    f"Freq={args.use_freq_domain}\n")
            f.write(f"FGW={args.fgw_weight}, Freq={args.freq_weight}, "
                    f"Struct={args.struct_weight}, SinoLoss={args.sino_weight}\n\n")

    # --- Training loop ---
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # 核心：每个 Virtual Epoch 开始前，让 Dataset 重新抽样一次
        dataset_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, dataloader_train, optimizer, device, epoch, args
        )

        ep_loss = train_stats['loss']
        train_losses.append(ep_loss)

        # Log
        log_parts = [f"Epoch {epoch:03d} | Total Loss: {ep_loss:.6f}"]
        for key in ['v_loss', 'fgw_loss', 'freq_loss', 'struct_loss', 'sino_loss']:
            if key in train_stats:
                log_parts.append(f"{key}: {train_stats[key]:.6f}")
        log_line = ", ".join(log_parts)

        with open(log_file_path, 'a') as f:
            f.write(log_line + "\n")

        print(f"  {log_line}")

        # --- Loss curve plot ---
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, label='Total Loss', color='blue', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value (Log Scale)')
        plt.title(f'TriDo-CNN Training Loss ({args.model_size})')
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'), dpi=150)
        plt.close()

        # --- Checkpoint saving ---
        is_save_epoch = False

        if epoch <= args.epochs - 50:
            if epoch % 50 == 0:
                is_save_epoch = True
        else:
            if epoch % 5 == 0:
                is_save_epoch = True

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
            print(f"  Saved checkpoint to {save_path} (including EMA)")

    # --- Done ---
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time / 3600:.2f} hours")

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, 'checkpoint-final.pth')
    torch.save({
        'model': model.net.state_dict(),
        'model_ema': model.get_ema_state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.epochs - 1,
        'args': args
    }, final_path)
    print(f"Final checkpoint saved to {final_path}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

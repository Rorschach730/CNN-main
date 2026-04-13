import argparse
import os
import time
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from util.pet_dataset_new import PETDenoisingDataset
from denoiser_cosine import Denoiser
from engine_jit import train_one_epoch
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('JiT PET Denoising', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=4, type=int, help='Gradient Accumulation steps')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    parser.add_argument('--P_mean', default=0.0, type=float)
    parser.add_argument('--P_std', default=1.2, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=0.001, type=float)
    parser.add_argument('--data_path', default='./processed_data_3d_osem', type=str)
    parser.add_argument('--output_dir', default='./output_dir_cosine', help='模型保存路径')
    parser.add_argument('--device', default='cuda', help='Device: GPU')
    parser.add_argument('--resume', default='', help='Resume checkpoint path')
    parser.add_argument('--warmup_epochs', type=int, default=20)

    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    return parser


def main(args):
    print(f"Start training on {args.device} with batch size {args.batch_size}, accum_iter {args.accum_iter}")
    device = torch.device(args.device)

    torch.set_float32_matmul_precision('high')

    dataset_train = PETDenoisingDataset(os.path.join(args.data_path, 'train'), img_size=args.img_size)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
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
        train_stats = train_one_epoch(model, dataloader_train, optimizer, device, epoch, args)

        ep_loss = train_stats['loss']
        train_losses.append(ep_loss)

        with open(log_file_path, 'a') as f:
            f.write(f"Epoch {epoch:03d} | Train Loss: {ep_loss:.6f}\n")

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, label='Composite Loss (MSE+L1+SSIM)', color='blue',
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

        if epoch % 5 == 0 or epoch == args.epochs - 1:
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
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from util.pet_dataset import PETDenoisingDataset
# 严密隔离两种物理拓扑的类导入
from denoiser import Denoiser as DenoiserBaseline
from denoiser_cosine import Denoiser as DenoiserCosine


def get_args_parser():
    parser = argparse.ArgumentParser('JiT OTF Sweet Point Dual Search', add_help=False)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='推理 Batch Size')
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    parser.add_argument('--P_mean', default=0.0, type=float)
    parser.add_argument('--P_std', default=1.2, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--data_path', default='./processed_data_sinogram/test', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    # [双轨搜索核心配置]
    parser.add_argument('--models', nargs='+', choices=['baseline', 'cosine', 'both'], default=['baseline'],
                        help='选择需要搜索的模型类型')

    # Baseline 的默认参数
    parser.add_argument('--dir_baseline', default='./output_dir', type=str)
    parser.add_argument('--epochs_baseline', nargs='+', type=int, default=[105, 110, 115, 120, 125, 130, 135, 140],
                        help='Baseline 模型的测试 Epoch 列表')

    # Cosine 的默认参数
    parser.add_argument('--dir_cosine', default='./output_dir_cosine', type=str)
    parser.add_argument('--epochs_cosine', nargs='+', type=int, default=[40, 45, 50, 55, 60, 65, 70, 75, 80],
                        help='Cosine 模型的测试 Epoch 列表')

    parser.add_argument('--steps', nargs='+', type=int, default=[1, 2], help='要测试的 Euler 积分步数列表')
    parser.add_argument('--use_roi', action='store_true', help='开启则仅计算红框 ROI 区域指标，默认计算全图')
    return parser


def inverse_normalize(tensor):
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()


def get_roi_bbox(img_raw, threshold_ratio=0.05, padding=4):
    threshold = img_raw.max() * threshold_ratio
    rows, cols = np.any(img_raw > threshold, axis=1), np.any(img_raw > threshold, axis=0)
    if not np.any(rows) or not np.any(cols): return 0, img_raw.shape[0], 0, img_raw.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return max(0, rmin - padding), min(img_raw.shape[0], rmax + padding), max(0, cmin - padding), min(img_raw.shape[1],
                                                                                                      cmax + padding)


def evaluate_single_config(model, dataloader, device, steps, use_roi):
    all_psnr, all_ssim = [], []

    with torch.no_grad():
        for targets, conditions in dataloader:
            targets = targets.to(device, non_blocking=True).to(torch.float32)
            conditions = conditions.to(device, non_blocking=True).to(torch.float32)

            generated = model.generate(conditions, steps=steps)

            gt_np = inverse_normalize(targets.squeeze(1))
            in_np = inverse_normalize(conditions.squeeze(1))
            pred_np = inverse_normalize(generated.squeeze(1))

            for i in range(gt_np.shape[0]):
                if gt_np[i].max() < 0.01: continue

                if use_roi:
                    rmin, rmax, cmin, cmax = get_roi_bbox(gt_np[i])
                    eval_gt = gt_np[i, rmin:rmax, cmin:cmax]
                    eval_pred = pred_np[i, rmin:rmax, cmin:cmax]
                else:
                    eval_gt = gt_np[i]
                    eval_pred = pred_np[i]

                all_psnr.append(compare_psnr(eval_gt, eval_pred, data_range=1.0))
                all_ssim.append(compare_ssim(eval_gt, eval_pred, data_range=1.0))

    return np.mean(all_psnr), np.mean(all_ssim)


def run_grid_search(model_name, ModelClass, ckpt_dir, epochs_list, steps_list, device, dataloader_test, args):
    print(f"\n" + "=" * 60)
    print(f"  [>] Launching Search for: {model_name.upper()}")
    print(f"  [>] Target Directory: {ckpt_dir}")
    print(f"  [>] Epochs to scan: {epochs_list}")
    print("=" * 60)

    # 仅实例化一次网络拓扑
    model = ModelClass(args).to(device)
    model.eval()

    results_matrix = []
    best_psnr = 0.0
    best_ssim = 0.0
    best_config = {}

    for epoch in epochs_list:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint-{epoch}.pth')
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint missing: {ckpt_path}, skipping...")
            continue

        print(f"\n[+] Injecting Weights from Epoch {epoch}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_ema', ckpt['model'])
        new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
        model.net.load_state_dict(new_state_dict, strict=False)

        for step in steps_list:
            print(f"    -> Evaluating Euler Steps = {step} ", end="")
            mean_psnr, mean_ssim = evaluate_single_config(model, dataloader_test, device, step, args.use_roi)
            print(f"... PSNR: {mean_psnr:.2f} | SSIM: {mean_ssim:.4f}")

            results_matrix.append({'epoch': epoch, 'steps': step, 'psnr': mean_psnr, 'ssim': mean_ssim})

            if mean_ssim > best_ssim:
                best_ssim = mean_ssim
                best_config = {'epoch': epoch, 'steps': step, 'psnr': mean_psnr, 'ssim': mean_ssim}

    return best_config, results_matrix


def main(args):
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    print(f"--- [Grid Search] Dual OTF Sweet Point Explorer ---")
    print(f"[*] Evaluation Mode: {'ROI' if args.use_roi else 'Whole Image'}")

    dataset_test = PETDenoisingDataset(args.data_path, img_size=args.img_size)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=4 if args.device == 'cuda' else 0,
        pin_memory=True if args.device == 'cuda' else False
    )

    tasks = []
    if 'both' in args.models or 'baseline' in args.models:
        tasks.append(('Baseline (No Cosine)', DenoiserBaseline, args.dir_baseline, args.epochs_baseline))
    if 'both' in args.models or 'cosine' in args.models:
        tasks.append(('Cosine (Directional)', DenoiserCosine, args.dir_cosine, args.epochs_cosine))

    all_best_configs = {}

    for name, ModelClass, c_dir, e_list in tasks:
        best_cfg, matrix = run_grid_search(name, ModelClass, c_dir, e_list, args.steps, device, dataloader_test, args)
        all_best_configs[name] = best_cfg

        if best_cfg:
            print(f"\n[{name}] SWEET POINT FOUND:")
            print(
                f"🏆 Epoch : {best_cfg['epoch']} | Steps: {best_cfg['steps']} | PSNR: {best_cfg['psnr']:.2f} | SSIM: {best_cfg['ssim']:.4f}")

    print("\n" + "=" * 60)
    print("               FINAL BATTLE SUMMARY               ")
    print("=" * 60)
    for name, cfg in all_best_configs.items():
        if cfg:
            print(
                f"🔹 {name.ljust(22)} -> Epoch {str(cfg['epoch']).ljust(3)} | Steps {cfg['steps']} | PSNR: {cfg['psnr']:.2f} dB | SSIM: {cfg['ssim']:.4f}")
        else:
            print(f"🔹 {name.ljust(22)} -> [!] No valid data.")
    print("=" * 60)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
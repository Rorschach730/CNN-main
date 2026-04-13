import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from util.pet_dataset_new import PETDenoisingDataset
from denoiser_cosine import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT PET Denoising Evaluation', add_help=False)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='推理 Batch Size')
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    parser.add_argument('--P_mean', default=0.0, type=float)
    parser.add_argument('--P_std', default=1.2, type=float)
    parser.add_argument('--accum_iter', default=4, type=int)

    parser.add_argument('--data_path', default='./processed_data_3d_osem/test', type=str)
    parser.add_argument('--checkpoint', default='./output_dir_cosine/checkpoint-65.pth', type=str)
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')

    parser.add_argument('--use_roi', default=True, action='store_true', help='是否仅在器官 ROI 计算指标')
    parser.add_argument('--steps', default=50, type=int, help='Heun 采样步数')
    return parser


def calculate_snr(clean, test):
    var_clean, var_noise = np.var(clean), np.var(clean - test)
    return float('inf') if var_noise == 0 else 10 * np.log10(var_clean / var_noise)


def get_roi_bbox(img_raw, threshold_ratio=0.05, padding=4):
    threshold = img_raw.max() * threshold_ratio
    rows, cols = np.any(img_raw > threshold, axis=1), np.any(img_raw > threshold, axis=0)
    if not np.any(rows) or not np.any(cols): return 0, img_raw.shape[0], 0, img_raw.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return max(0, rmin - padding), min(img_raw.shape[0], rmax + padding), max(0, cmin - padding), min(img_raw.shape[1],
                                                                                                      cmax + padding)


def main(args):
    eval_mode = "ROI" if args.use_roi else "Whole Image"
    print(f"--- [Evaluation] Initializing Metrics on {args.device} (Mode: {eval_mode}) ---")
    device = torch.device(args.device)

    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    dataset_test = PETDenoisingDataset(args.data_path, img_size=args.img_size)

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if args.device == 'cuda' else 0,
        pin_memory=True if args.device == 'cuda' else False
    )

    model = Denoiser(args).to(device)

    print(f"[*] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    state_dict = ckpt.get('model_ema', ckpt['model'])
    new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.net.load_state_dict(new_state_dict, strict=False)
    model.eval()

    all_psnr_in, all_ssim_in, all_snr_in = [], [], []
    all_psnr_out, all_ssim_out, all_snr_out = [], [], []

    print(f"[*] Starting batched inference (Heun steps={args.steps})...")
    with torch.no_grad():
        for batch_idx, (targets, conditions) in enumerate(tqdm(dataloader_test, desc="Evaluating")):
            targets = targets.to(device, non_blocking=True).to(torch.float32)
            conditions = conditions.to(device, non_blocking=True).to(torch.float32)

            generated = model.generate(conditions, steps=args.steps)

            gt_np = np.clip(targets.squeeze(1).cpu().numpy(), 0.0, 1.0)
            in_np = np.clip(conditions[:, 1, :, :].cpu().numpy(), 0.0, 1.0)
            pred_np = np.clip(generated.squeeze(1).cpu().numpy(), 0.0, 1.0)

            for i in range(gt_np.shape[0]):
                if gt_np[i].max() < 0.01: continue

                if args.use_roi:
                    rmin, rmax, cmin, cmax = get_roi_bbox(gt_np[i])
                    eval_gt = gt_np[i, rmin:rmax, cmin:cmax]
                    eval_in = in_np[i, rmin:rmax, cmin:cmax]
                    eval_pred = pred_np[i, rmin:rmax, cmin:cmax]
                else:
                    eval_gt = gt_np[i]
                    eval_in = in_np[i]
                    eval_pred = pred_np[i]

                p_in = compare_psnr(eval_gt, eval_in, data_range=1.0)
                s_in = compare_ssim(eval_gt, eval_in, data_range=1.0)
                snr_in_val = calculate_snr(eval_gt, eval_in)
                all_psnr_in.append(p_in)
                all_ssim_in.append(s_in)
                all_snr_in.append(snr_in_val)

                p_out = compare_psnr(eval_gt, eval_pred, data_range=1.0)
                s_out = compare_ssim(eval_gt, eval_pred, data_range=1.0)
                snr_out_val = calculate_snr(eval_gt, eval_pred)
                all_psnr_out.append(p_out)
                all_ssim_out.append(s_out)
                all_snr_out.append(snr_out_val)

    mean_psnr_in, std_psnr_in = np.mean(all_psnr_in), np.std(all_psnr_in)
    mean_ssim_in, std_ssim_in = np.mean(all_ssim_in), np.std(all_ssim_in)
    mean_snr_in, std_snr_in = np.mean(all_snr_in), np.std(all_snr_in)

    mean_psnr_out, std_psnr_out = np.mean(all_psnr_out), np.std(all_psnr_out)
    mean_ssim_out, std_ssim_out = np.mean(all_ssim_out), np.std(all_ssim_out)
    mean_snr_out, std_snr_out = np.mean(all_snr_out), np.std(all_snr_out)

    diff_psnr = mean_psnr_out - mean_psnr_in
    diff_ssim = mean_ssim_out - mean_ssim_in
    diff_snr = mean_snr_out - mean_snr_in

    print("\n" + "=" * 65 + f"\n      FINAL 2.5D HEUN METRICS ({eval_mode.upper()})     \n" + "=" * 65)
    print(f"Total Evaluated Valid Slices: {len(all_psnr_out)}")
    print(f"--- Baseline (Noisy Input) ---")
    print(f"PSNR : {mean_psnr_in:.2f} ± {std_psnr_in:.2f} dB")
    print(f"SSIM : {mean_ssim_in:.4f} ± {std_ssim_in:.4f}")
    print(f"SNR  : {mean_snr_in:.2f} ± {std_snr_in:.2f} dB")
    print(f"--- JiT-Heun 50 Denoised Output ---")
    print(f"PSNR : {mean_psnr_out:.2f} ± {std_psnr_out:.2f} dB  ( {diff_psnr:+.2f} dB )")
    print(f"SSIM : {mean_ssim_out:.4f} ± {std_ssim_out:.4f}  ( {diff_ssim:+.4f} )")
    print(f"SNR  : {mean_snr_out:.2f} ± {std_snr_out:.2f} dB  ( {diff_snr:+.2f} dB )")
    print("=" * 65)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
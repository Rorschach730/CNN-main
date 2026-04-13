import argparse
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from concurrent.futures import ProcessPoolExecutor, as_completed

from util.pet_dataset import PETDenoisingDataset
from util.prepare_data_sinogram import SimConfig


def get_args_parser():
    parser = argparse.ArgumentParser('TV-OSEM Baseline Evaluation', add_help=False)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--data_path', default='./processed_data_sinogram/test', type=str)
    parser.add_argument('--save_dir', default='./results_vis_osem', type=str)
    parser.add_argument('--use_roi', action='store_true', help='是否仅在器官 ROI 区域计算指标')
    parser.add_argument('--iterations', default=2, type=int, help='OSEM 迭代次数')
    parser.add_argument('--subsets', default=12, type=int, help='OSEM 子集数量')

    # [物理升维] 全变分正则化超参数
    parser.add_argument('--beta', default=0.07, type=float, help='TV 正则化强度 (越大越平滑)')

    parser.add_argument('--num_samples', default=-1, type=int, help='评估的切片数量 (-1 为全部)')
    parser.add_argument('--workers', default=14, type=int, help='启用的 CPU 核心数')
    return parser


def inverse_normalize(tensor):
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()


def calculate_snr(clean, test):
    var_clean, var_noise = np.var(clean), np.var(clean - test)
    return float('inf') if var_noise == 0 else 10 * np.log10(var_clean / var_noise)


def robust_windowing(img_gt):
    valid_pixels = img_gt[img_gt > 0.01]
    return (0.0, np.percentile(valid_pixels, 99)) if len(valid_pixels) > 0 else (0.0, 1.0)


def get_roi_bbox(img_raw, threshold_ratio=0.05, padding=4):
    threshold = img_raw.max() * threshold_ratio
    rows, cols = np.any(img_raw > threshold, axis=1), np.any(img_raw > threshold, axis=0)
    if not np.any(rows) or not np.any(cols): return 0, img_raw.shape[0], 0, img_raw.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return max(0, rmin - padding), min(img_raw.shape[0], rmax + padding), max(0, cmin - padding), min(img_raw.shape[1],
                                                                                                      cmax + padding)


def draw_and_save_1x4(img_gt, img_in, img_pred, metrics_in, metrics_out, save_path, title_suffix, beta):
    vmin, vmax = robust_windowing(img_gt)
    err_out = np.abs(img_gt - img_pred)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_in, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Noisy FBP Input\nPSNR: {metrics_in[0]:.2f} | SSIM: {metrics_in[1]:.3f}")
    axes[0].axis('off')

    axes[1].imshow(img_pred, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"TV-OSEM (β={beta})\nPSNR: {metrics_out[0]:.2f} | SSIM: {metrics_out[1]:.3f}")
    axes[1].axis('off')

    axes[2].imshow(img_gt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title("Ground Truth (Clean)")
    axes[2].axis('off')

    im3 = axes[3].imshow(err_out, cmap='jet', vmin=0, vmax=vmax * 0.5)
    axes[3].set_title("Error Heatmap (|TV-OSEM - GT|)")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def osem_reconstruction(sinogram_noisy, theta, iterations, subsets, img_size, beta):
    recon = np.ones((img_size, img_size), dtype=np.float32)
    num_angles = len(theta)

    for i in range(iterations):
        for s in range(subsets):
            idx = np.arange(s, num_angles, subsets)
            sub_theta = theta[idx]
            sub_sino = sinogram_noisy[:, idx]

            ones_sino = np.ones_like(sub_sino)
            sens_img = iradon(ones_sino, theta=sub_theta, circle=False, filter_name=None, output_size=img_size)
            sens_img[sens_img <= 0] = 1e-8

            proj = radon(recon, theta=sub_theta, circle=False)
            proj[proj <= 0] = 1e-8

            ratio = sub_sino / proj
            bp_ratio = iradon(ratio, theta=sub_theta, circle=False, filter_name=None, output_size=img_size)

            # [正向步] 泊松似然乘法更新
            recon = recon * (bp_ratio / sens_img)

        # [反向步] 注入 Chambolle 全变分投影算子 (Proximal Mapping)
        if beta > 0:
            recon = denoise_tv_chambolle(recon, weight=beta)

    return recon


def process_single_slice(pack):
    batch_idx, gt_np, in_np, use_roi, iterations, subsets, img_size, save_dir, beta = pack

    theta = np.linspace(0., 180., img_size, endpoint=False)
    sinogram_ideal = radon(gt_np, theta=theta, circle=False)
    current_sum = sinogram_ideal.sum()
    if current_sum == 0:
        return None

    scale_factor = SimConfig.TOTAL_COUNTS_HIGH / current_sum
    sinogram_counts_high = sinogram_ideal * scale_factor
    sinogram_counts_low = sinogram_counts_high / SimConfig.DRF

    sinogram_noisy_counts = np.random.poisson(sinogram_counts_low).astype(np.float32)

    # 调用搭载了 TV 引擎的 OSEM
    osem_recon = osem_reconstruction(sinogram_noisy_counts, theta, iterations=iterations, subsets=subsets,
                                     img_size=img_size, beta=beta)

    osem_recon = osem_recon / scale_factor * SimConfig.DRF
    pred_np = np.clip(osem_recon, 0.0, 1.0)

    if use_roi:
        rmin, rmax, cmin, cmax = get_roi_bbox(gt_np)
        eval_gt = gt_np[rmin:rmax, cmin:cmax]
        eval_in = in_np[rmin:rmax, cmin:cmax]
        eval_pred = pred_np[rmin:rmax, cmin:cmax]
    else:
        eval_gt = gt_np
        eval_in = in_np
        eval_pred = pred_np

    p_in = compare_psnr(eval_gt, eval_in, data_range=1.0)
    s_in = compare_ssim(eval_gt, eval_in, data_range=1.0)
    snr_in_val = calculate_snr(eval_gt, eval_in)

    p_out = compare_psnr(eval_gt, eval_pred, data_range=1.0)
    s_out = compare_ssim(eval_gt, eval_pred, data_range=1.0)
    snr_out_val = calculate_snr(eval_gt, eval_pred)

    if batch_idx < 5:
        save_path = os.path.join(save_dir, f"tv_osem_eval_slice_{batch_idx}.png")
        draw_and_save_1x4(gt_np, in_np, pred_np, (p_in, s_in, snr_in_val), (p_out, s_out, snr_out_val), save_path, "",
                          beta)

    return {
        'in': (p_in, s_in, snr_in_val),
        'out': (p_out, s_out, snr_out_val)
    }


def main(args):
    eval_mode = "ROI" if args.use_roi else "Whole Image"
    print(f"--- [Baseline] Initializing TV-OSEM Metrics (Mode: {eval_mode}, β={args.beta}) ---")
    print(f"[*] Hardware: Utilizing {args.workers} CPU Threads")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    dataset_test = PETDenoisingDataset(args.data_path, img_size=args.img_size)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

    print("[*] Pre-fetching and normalizing strictly aligned slices...")
    tasks = []
    for batch_idx, (targets, conditions) in enumerate(dataloader_test):
        gt_np = inverse_normalize(targets.squeeze(1))[0]
        in_np = inverse_normalize(conditions.squeeze(1))[0]

        if gt_np.max() < 0.01: continue
        tasks.append(
            (batch_idx, gt_np, in_np, args.use_roi, args.iterations, args.subsets, args.img_size, args.save_dir,
             args.beta))

        if args.num_samples != -1 and len(tasks) >= args.num_samples:
            break

    all_psnr_in, all_ssim_in, all_snr_in = [], [], []
    all_psnr_out, all_ssim_out, all_snr_out = [], [], []

    print("[*] Launching TV-OSEM Multiprocessing Engine...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_slice, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"TV-OSEM (i={args.iterations}, s={args.subsets}, β={args.beta})"):
            res = future.result()
            if res is not None:
                all_psnr_in.append(res['in'][0])
                all_ssim_in.append(res['in'][1])
                all_snr_in.append(res['in'][2])

                all_psnr_out.append(res['out'][0])
                all_ssim_out.append(res['out'][1])
                all_snr_out.append(res['out'][2])

    mean_psnr_in, std_psnr_in = np.mean(all_psnr_in), np.std(all_psnr_in)
    mean_ssim_in, std_ssim_in = np.mean(all_ssim_in), np.std(all_ssim_in)
    mean_snr_in, std_snr_in = np.mean(all_snr_in), np.std(all_snr_in)

    mean_psnr_out, std_psnr_out = np.mean(all_psnr_out), np.std(all_psnr_out)
    mean_ssim_out, std_ssim_out = np.mean(all_ssim_out), np.std(all_ssim_out)
    mean_snr_out, std_snr_out = np.mean(all_snr_out), np.std(all_snr_out)

    diff_psnr = mean_psnr_out - mean_psnr_in
    diff_ssim = mean_ssim_out - mean_ssim_in
    diff_snr = mean_snr_out - mean_snr_in

    print("\n" + "=" * 65 + f"\n   FINAL TV-OSEM METRICS ({eval_mode.upper()})   \n" + "=" * 65)
    print(f"Total Evaluated Valid Slices: {len(all_psnr_out)}")
    print(f"--- Baseline (Noisy FBP) ---")
    print(f"PSNR : {mean_psnr_in:.2f} ± {std_psnr_in:.2f} dB")
    print(f"SSIM : {mean_ssim_in:.4f} ± {std_ssim_in:.4f}")
    print(f"SNR  : {mean_snr_in:.2f} ± {std_snr_in:.2f} dB")
    print(f"--- TV-OSEM Reconstruction ({args.iterations} iters, {args.subsets} subsets, β={args.beta}) ---")
    print(f"PSNR : {mean_psnr_out:.2f} ± {std_psnr_out:.2f} dB  ( {diff_psnr:+.2f} dB )")
    print(f"SSIM : {mean_ssim_out:.4f} ± {std_ssim_out:.4f}  ( {diff_ssim:+.4f} )")
    print(f"SNR  : {mean_snr_out:.2f} ± {std_snr_out:.2f} dB  ( {diff_snr:+.2f} dB )")
    print("=" * 65)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
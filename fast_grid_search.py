import argparse
import os
import torch
import numpy as np
import random
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from util.pet_dataset_new import PETDenoisingDataset
from denoiser_new import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT Full Validation Grid Search (CWS)', add_help=False)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)

    # 物理环境对齐
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--cond_drop_prob', default=0.1, type=float)
    parser.add_argument('--P_mean', default=-0.5, type=float)
    parser.add_argument('--P_std', default=1.2, type=float)

    parser.add_argument('--data_path', default='./processed_data_3d_osem/val', type=str)
    parser.add_argument('--ckpt_dir', default='./output_dir_new', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--epochs', nargs='+', type=int,
                        default=list(range(300, 400, 5)) + [399], help='扫描的 Epoch 列表')
    parser.add_argument('--cfg', type=float, default=0.6, help='刚性锁定的 CFG 值')
    parser.add_argument('--steps', default=50, type=int, help='Heun 积分步数')
    parser.add_argument('--seed', default=42, type=int)

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def calculate_snr(clean, test):
    var_clean = np.var(clean)
    var_noise = np.var(clean - test)
    if var_noise == 0: return float('inf')
    return 10 * np.log10(var_clean / var_noise)


def calculate_clinical_metrics(gt_roi, pred_roi):
    eps = 1e-8
    max_gt = gt_roi.max()
    max_pred = pred_roi.max()
    bias_suvmax = ((max_pred - max_gt) / (max_gt + eps)) * 100.0

    mean_gt = gt_roi.mean()
    mean_pred = pred_roi.mean()
    mape_suvmean = (np.abs(mean_pred - mean_gt) / (mean_gt + eps)) * 100.0

    thresh95 = np.percentile(gt_roi, 95)
    hotspot_mask = gt_roi >= thresh95
    mean_hotspot_gt = gt_roi[hotspot_mask].mean()
    mean_hotspot_pred = pred_roi[hotspot_mask].mean()
    cr = (mean_hotspot_pred / (mean_hotspot_gt + eps)) * 100.0

    return bias_suvmax, mape_suvmean, cr


def get_roi_bbox(img_raw, threshold_ratio=0.05, padding=4):
    threshold = img_raw.max() * threshold_ratio
    rows, cols = np.any(img_raw > threshold, axis=1), np.any(img_raw > threshold, axis=0)
    if not np.any(rows) or not np.any(cols): return 0, img_raw.shape[0], 0, img_raw.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return max(0, rmin - padding), min(img_raw.shape[0], rmax + padding), max(0, cmin - padding), min(img_raw.shape[1],
                                                                                                      cmax + padding)


def compute_cws(bias, cr, mape, ssim):
    """
    核医学临床综合评分系统 (Clinical Weighted Score)
    基础分 100，根据各项指标的临床致命性执行扣分惩罚。
    """
    penalty_bias = 2.0 * abs(bias)
    penalty_cr = 1.5 * abs(100.0 - cr)
    penalty_mape = 1.0 * mape
    penalty_ssim = 0.5 * 100.0 * (1.0 - ssim)

    score = 100.0 - penalty_bias - penalty_cr - penalty_mape - penalty_ssim
    return score


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda': torch.set_float32_matmul_precision('high')

    print("=" * 80)
    print(f"[*] FULL VALIDATION: Clinical Weighted Score (CWS) Analysis")
    print(f"[*] CFG Locked    : {args.cfg}")
    print(f"[*] CWS Formula   : 100 - (2.0*|Bias|) - (1.5*|100-CR|) - (1.0*MAPE) - (0.5*100*(1-SSIM))")
    print("=" * 80)

    # 强制全量挂载，切除子集采样
    full_dataset = PETDenoisingDataset(args.data_path, img_size=args.img_size)
    print(f"[*] Loading Full Validation Set: {len(full_dataset)} physical slices.")

    dataloader = torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4 if args.device == 'cuda' else 0, pin_memory=True
    )

    model = Denoiser(args).to(device)
    model.eval()

    results_matrix = []
    baseline_metrics = {'cr': [], 'ssim': [], 'bias': [], 'mape': [], 'snr': [], 'psnr': []}

    print("[*] Computing Noisy Input Baseline over Full Dataset...")
    with torch.no_grad():
        for targets, conditions in dataloader:
            gt_raw = targets.squeeze(1).cpu().numpy()
            in_raw = conditions.squeeze(1).cpu().numpy()
            gt_01 = np.clip(gt_raw, 0.0, 1.0)
            in_01 = np.clip(in_raw, 0.0, 1.0)

            for i in range(gt_raw.shape[0]):
                if gt_raw[i].max() < 0.05: continue
                rmin, rmax, cmin, cmax = get_roi_bbox(gt_01[i])

                b, m, c = calculate_clinical_metrics(gt_raw[i, rmin:rmax, cmin:cmax], in_raw[i, rmin:rmax, cmin:cmax])
                p = compare_psnr(gt_01[i, rmin:rmax, cmin:cmax], in_01[i, rmin:rmax, cmin:cmax], data_range=1.0)
                s = compare_ssim(gt_01[i, rmin:rmax, cmin:cmax], in_01[i, rmin:rmax, cmin:cmax], data_range=1.0)
                n = calculate_snr(gt_01[i, rmin:rmax, cmin:cmax], in_01[i, rmin:rmax, cmin:cmax])

                baseline_metrics['cr'].append(c);
                baseline_metrics['ssim'].append(s)
                baseline_metrics['bias'].append(b);
                baseline_metrics['mape'].append(m)
                baseline_metrics['psnr'].append(p);
                baseline_metrics['snr'].append(n)

    base_avg = {k: np.mean(v) for k, v in baseline_metrics.items()}
    base_cws = compute_cws(base_avg['bias'], base_avg['cr'], base_avg['mape'], base_avg['ssim'])
    print(
        f"    [Baseline] CWS: {base_cws:.2f} | CR: {base_avg['cr']:.1f}% | SSIM: {base_avg['ssim']:.4f} | Bias: {base_avg['bias']:.1f}% | MAPE: {base_avg['mape']:.1f}%")

    print("\n[*] Commencing Epoch Sweep...")
    for epoch in args.epochs:
        ckpt_path = os.path.join(args.ckpt_dir, f'checkpoint-{epoch}.pth')
        if not os.path.exists(ckpt_path): continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.net.load_state_dict({k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in
                                   ckpt.get('model_ema', ckpt['model']).items()}, strict=False)

        m_cr, m_ssim, m_bias, m_mape, m_psnr, m_snr = [], [], [], [], [], []

        with torch.no_grad():
            for targets, conditions in dataloader:
                conditions_t = conditions.to(device).to(torch.float32)
                generated = model.generate(conditions_t, steps=args.steps, cfg_scale=args.cfg)

                gt_raw = targets.squeeze(1).cpu().numpy()
                pred_raw = generated.squeeze(1).cpu().numpy()
                gt_01 = np.clip(gt_raw, 0.0, 1.0)
                pred_01 = np.clip(pred_raw, 0.0, 1.0)

                for i in range(gt_raw.shape[0]):
                    if gt_raw[i].max() < 0.05: continue
                    rmin, rmax, cmin, cmax = get_roi_bbox(gt_01[i])

                    b, m, c = calculate_clinical_metrics(gt_raw[i, rmin:rmax, cmin:cmax],
                                                         pred_raw[i, rmin:rmax, cmin:cmax])
                    p = compare_psnr(gt_01[i, rmin:rmax, cmin:cmax], pred_01[i, rmin:rmax, cmin:cmax], data_range=1.0)
                    s = compare_ssim(gt_01[i, rmin:rmax, cmin:cmax], pred_01[i, rmin:rmax, cmin:cmax], data_range=1.0)
                    n = calculate_snr(gt_01[i, rmin:rmax, cmin:cmax], pred_01[i, rmin:rmax, cmin:cmax])

                    m_cr.append(c);
                    m_ssim.append(s);
                    m_bias.append(b)
                    m_mape.append(m);
                    m_psnr.append(p);
                    m_snr.append(n)

        avg_cr, avg_ssim, avg_bias = np.mean(m_cr), np.mean(m_ssim), np.mean(m_bias)
        avg_mape, avg_psnr, avg_snr = np.mean(m_mape), np.mean(m_psnr), np.mean(m_snr)

        cws = compute_cws(avg_bias, avg_cr, avg_mape, avg_ssim)

        res = {
            'epoch': epoch, 'cws': cws,
            'cr': avg_cr, 'ssim': avg_ssim, 'bias': avg_bias,
            'mape': avg_mape, 'psnr': avg_psnr, 'snr': avg_snr
        }
        results_matrix.append(res)
        print(
            f"    [Epoch {epoch:<3}] CWS: {cws:>6.2f} | CR: {avg_cr:.1f}% | Bias: {avg_bias:>4.1f}% | MAPE: {avg_mape:.1f}% | SSIM: {avg_ssim:.4f}")

    best_config = max(results_matrix, key=lambda x: x['cws'])

    print("\n" + "=" * 80)
    print("               FINAL CLINICAL SWEET SPOT (FULL VAL)               ")
    print("=" * 80)
    print(
        f"🏆 Epoch : {best_config['epoch']}  |  CWS Score : {best_config['cws']:.2f} (Gain: {best_config['cws'] - base_cws:+.2f})")
    print(f"   [SUVmax Bias]       : {best_config['bias']:.1f}%    (Base: {base_avg['bias']:.1f}%)")
    print(f"   [Contrast Recovery] : {best_config['cr']:.1f}%   (Gain: {best_config['cr'] - base_avg['cr']:+.1f}%)")
    print(f"   [SUVmean MAPE]      : {best_config['mape']:.1f}%    (Base: {base_avg['mape']:.1f}%)")
    print(f"   [ROI SSIM]          : {best_config['ssim']:.4f}   (Gain: {best_config['ssim'] - base_avg['ssim']:+.4f})")
    print("=" * 80)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
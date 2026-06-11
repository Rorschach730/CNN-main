"""
TriDo-JiT 批量评估脚本 (Full Test Set Evaluation)
==================================================
基于 test_evaluation_ud.py 重写，适配 trido_ud 三域架构。

核心适配:
  - 模型: TriDoDenoiser (替代 Denoiser)
  - 数据: TriDoPETDataset → (target, condition, body_part)
  - 推理: model.generate(condition, body_part, steps, cfg_scale)
  - 输出: PSNR/SSIM/SNR/MSE/SUVmax Bias + 可视化 + ROI .npy

用法:
    python trido_ud/evaluate_trido_fixed.py

依赖:
    pip install scikit-image tqdm matplotlib numpy torch
"""

import os
import re
import csv
import time
import argparse
import multiprocessing as mp
import queue

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ── 三域模块 ──
from trido_ud.denoiser_trido import TriDoDenoiser
from trido_ud.pet_dataset_trido import TriDoPETDataset

# ── 开启 TF32 加速 ──
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════
# 指标计算工具
# ═══════════════════════════════════════════════════════════════

def calculate_snr(clean: np.ndarray, test: np.ndarray) -> float:
    var_clean = np.var(clean)
    var_noise = np.var(clean - test)
    if var_noise == 0:
        return float('inf')
    return 10 * np.log10(var_clean / var_noise)


def calculate_suv_bias(gt_roi: np.ndarray, pred_roi: np.ndarray) -> float:
    """SUVmax Bias (98% percentile)"""
    eps = 1e-8
    max_gt = np.percentile(gt_roi, 98)
    max_pred = np.percentile(pred_roi, 98)
    return ((max_pred - max_gt) / (max_gt + eps)) * 100.0


def robust_windowing(img_gt: np.ndarray):
    """Robust display window [0, p99]"""
    valid = img_gt[img_gt > 0.01]
    if len(valid) == 0:
        return 0.0, 1.0
    return 0.0, np.percentile(valid, 99)


def get_roi_bbox(img_raw: np.ndarray, threshold_ratio=0.05, padding=4):
    """Get ROI bounding box"""
    threshold = img_raw.max() * threshold_ratio
    rows = np.any(img_raw > threshold, axis=1)
    cols = np.any(img_raw > threshold, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, img_raw.shape[0], 0, img_raw.shape[1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (max(0, rmin - padding),
            min(img_raw.shape[0], rmax + padding),
            max(0, cmin - padding),
            min(img_raw.shape[1], cmax + padding))


def extract_dose_from_name(filename: str) -> float:
    """Try to extract dose ratio from filename (_D{dose}_ or _DR{ratio}_)"""
    # Pattern: _D100_ → 1/100 = 0.01
    match = re.search(r'_D(\d+)_', filename)
    if match:
        return 1.0 / float(match.group(1))
    # Fallback: unknown
    return float('nan')


# ═══════════════════════════════════════════════════════════════
# 后台 Worker — ROI 计算 + CSV 写入 + 可视化 + .npy 落盘
# ═══════════════════════════════════════════════════════════════

def background_worker(task_queue, result_queue, output_vis_dir, output_npy_dir):
    while True:
        try:
            task = task_queue.get(timeout=3)
            if task is None:
                break

            base_name, z_str, dose_val, body_part, gt_raw, in_raw, out_raw = task

            # ── ROI 区域 ──
            rmin, rmax, cmin, cmax = get_roi_bbox(gt_raw)
            if (rmax - rmin) <= 0 or (cmax - cmin) <= 0:
                continue

            roi_gt = gt_raw[rmin:rmax, cmin:cmax]
            roi_in = in_raw[rmin:rmax, cmin:cmax]
            roi_out = out_raw[rmin:rmax, cmin:cmax]

            dyn_range = roi_gt.max() - roi_gt.min()
            if dyn_range == 0:
                dyn_range = 1.0

            # ── 指标计算 ──
            psnr_in  = compare_psnr(roi_gt, roi_in, data_range=dyn_range)
            ssim_in  = compare_ssim(roi_gt, roi_in, data_range=dyn_range)
            snr_in   = calculate_snr(roi_gt, roi_in)
            mse_in   = np.mean((roi_gt - roi_in) ** 2)
            bias_in  = calculate_suv_bias(roi_gt, roi_in)

            psnr_out = compare_psnr(roi_gt, roi_out, data_range=dyn_range)
            ssim_out = compare_ssim(roi_gt, roi_out, data_range=dyn_range)
            snr_out  = calculate_snr(roi_gt, roi_out)
            mse_out  = np.mean((roi_gt - roi_out) ** 2)
            bias_out = calculate_suv_bias(roi_gt, roi_out)

            bp_name = {0: "Brain", 1: "Chest", 2: "Abdomen"}.get(body_part, str(body_part))

            csv_row = [
                base_name, z_str, bp_name,
                f"{dose_val:.4f}" if not np.isnan(dose_val) else "N/A",
                f"{psnr_in:.4f}", f"{ssim_in:.4f}", f"{snr_in:.4f}",
                f"{mse_in:.6f}", f"{bias_in:.4f}",
                f"{psnr_out:.4f}", f"{ssim_out:.4f}", f"{snr_out:.4f}",
                f"{mse_out:.6f}", f"{bias_out:.4f}",
            ]
            result_queue.put(csv_row)

            # ── 落盘 ROI .npy ──
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_gt.npy"), roi_gt)
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_in.npy"), roi_in)
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_out.npy"), roi_out)

            # ── 可视化 ──
            vmin, vmax = robust_windowing(gt_raw)
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            for ax, img, title in zip(
                axes,
                [in_raw, out_raw, gt_raw],
                [f"Low Dose Input\n(Body: {bp_name})",
                 f"TriDo-JiT Denoised\n(PSNR: {psnr_out:.2f})",
                 "Full Dose Target"]
            ):
                ax.imshow(np.clip(img, 0, vmax), cmap='gray', vmin=vmin, vmax=vmax)
                ax.add_patch(plt.Rectangle(
                    (cmin, rmin), cmax - cmin, rmax - rmin,
                    fill=False, edgecolor='red', linewidth=1))
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            plt.tight_layout()
            fig.savefig(os.path.join(output_vis_dir, f"{base_name}.png"), dpi=100)
            plt.close(fig)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker Error] {e}")


# 【核心修复】：将 collate_fn 提取到全局作用域，确保 Windows 下多进程可以序列化它
def collate_fn(batch):
    """collate: 每个 sample = (target, condition, body_part)"""
    targets = torch.stack([s[0] for s in batch])
    conditions = torch.stack([s[1] for s in batch])
    body_parts = torch.tensor([s[2] for s in batch], dtype=torch.long)
    return targets, conditions, body_parts


# ═══════════════════════════════════════════════════════════════
# 主评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_trido(args):
    mp.set_start_method('spawn', force=True)

    device = torch.device(args.device)

    # ── 路径 ──
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    npy_dir = os.path.join(args.output_dir, "npy")
    csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    print(f"╔══════════════════════════════════════════╗")
    print(f"║   TriDo-JiT 批量评估 (Full Test Set)      ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"  Checkpoint : {args.ckpt_path}")
    print(f"  Test Data  : {args.data_path}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Device     : {device}")
    print(f"  Steps/CFG  : {args.nfe} / {args.cfg_scale}")

    # ── 加载模型 ──
    print(f"\n[*] 初始化 TriDoDenoiser (model_size={args.model_size})...")
    model = TriDoDenoiser(args).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if 'model_ema' in ckpt:
        print("[*] 加载 EMA 平滑权重")
        try:
            model.load_ema_state_dict(ckpt['model_ema'])
            # 【重要】确保将 EMA 参数复制到网络主体
            model.net.load_state_dict(ckpt['model_ema'], strict=False)
        except Exception as e:
            print(f"[!] 加载 EMA 报错，回退到常规权重... Error: {e}")
            model.net.load_state_dict(ckpt['model'], strict=False)
    elif 'model' in ckpt:
        model.net.load_state_dict(ckpt['model'], strict=False)
        print("[*] 加载常规权重")
    else:
        model.net.load_state_dict(ckpt, strict=False)
    model.eval()

    # ── 加载数据 ──
    print(f"\n[*] 加载测试集...")
    dataset = TriDoPETDataset(args.data_path, img_size=args.img_size, virtual_epoch_ratio=1.0)
    total_samples = len(dataset)
    if total_samples == 0:
        print("[!] 测试集为空，退出")
        return
    print(f"  共 {total_samples} 个测试样本")

    # ── 写 CSV 头 ──
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "Slice_Z", "Body_Part", "Dose_Ratio",
            "PSNR_In", "SSIM_In", "SNR_In", "MSE_In", "Bias_In(%)",
            "PSNR_Out", "SSIM_Out", "SNR_Out", "MSE_Out", "Bias_Out(%)"
        ])

    # ── DataLoader ──
    batch_size = min(args.batch_size, total_samples)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn, # 引用全局作用域的 collate_fn
    )

    # ── 后台 Worker 进程 ──
    num_procs = min(mp.cpu_count() - 1, args.workers)
    task_queue = mp.Queue(maxsize=200)
    result_queue = mp.Queue()
    workers = []
    for _ in range(num_procs):
        p = mp.Process(target=background_worker, args=(task_queue, result_queue, vis_dir, npy_dir))
        p.start()
        workers.append(p)
    print(f"\n[*] 已启动 {num_procs} 个后台处理进程，主进程开始 GPU 推理...")

    # ── 推理循环 ──
    sample_idx = 0
    t0 = time.time()

    for targets, conditions, body_parts in tqdm(dataloader, desc="GPU Inference"):
        targets = targets.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        body_parts = body_parts.to(device, non_blocking=True)
        B = targets.size(0)

        # ── GPU 推理 ──
        t_gpu_start = time.time()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=(device.type == 'cuda')
        ):
            # 自适应推理: 根据噪声水平自动调整 CFG + NFE
            if hasattr(model, 'generate_adaptive'):
                outputs = model.generate_adaptive(
                    conditions, body_parts,
                    base_steps=args.nfe, base_cfg=args.cfg_scale
                )
            else:
                outputs = model.generate(
                    conditions, body_parts,
                    steps=args.nfe, cfg_scale=args.cfg_scale
                )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_gpu = time.time() - t_gpu_start

        # ── 转 CPU ──
        out_np = outputs[:, 0, :, :].cpu().numpy()
        gt_np  = targets[:, 0, :, :].cpu().numpy()
        in_np  = conditions[:, 0, :, :].cpu().numpy()
        bp_np  = body_parts.cpu().numpy()

        # ── 逐样本投喂后台队列 ──
        for i in range(B):
            if gt_np[i].max() < 0.05:
                sample_idx += 1
                continue

            # 修改这部分避免 dataset.current_epoch_samples 越界
            if sample_idx < len(dataset.current_epoch_samples):
                file_path = dataset.current_epoch_samples[sample_idx]
                base_name = os.path.basename(file_path).replace('.pt', '')
                z_match = re.search(r'_Z(\d{4})', base_name)
                z_str = z_match.group(1) if z_match else f"{sample_idx:04d}"
                dose_val = extract_dose_from_name(base_name)
            else:
                base_name = f"sample_{sample_idx:06d}"
                z_str = "0000"
                dose_val = float('nan')

            task_queue.put((
                base_name, z_str, dose_val, int(bp_np[i]),
                gt_np[i], in_np[i], out_np[i]
            ))
            sample_idx += 1

        # ── 阶段性刷新 CSV ──
        batch_rows = []
        while not result_queue.empty():
            batch_rows.append(result_queue.get())
        if batch_rows:
            with open(csv_path, mode='a', newline='') as f:
                csv.writer(f).writerows(batch_rows)

        t_total = time.time() - t0
        t_data = t_total - t_gpu
        samples_done = sample_idx
        eta = (t_total / samples_done) * (total_samples - samples_done) if samples_done > 0 else 0
        print(f"\n[探针] Sample {samples_done}/{total_samples} | "
              f"GPU: {t_gpu:.1f}s/{B}imgs ({B/t_gpu:.0f} fps) | "
              f"Total: {t_total:.1f}s | ETA: {eta:.1f}s")
        t0 = time.time()

    # ── 收尾 ──
    for _ in range(num_procs):
        task_queue.put(None)
    for p in workers:
        p.join()

    remaining = []
    while not result_queue.empty():
        remaining.append(result_queue.get())
    if remaining:
        with open(csv_path, mode='a', newline='') as f:
            csv.writer(f).writerows(remaining)

    # ── 汇总统计 ──
    print(f"\n{'='*60}")
    print(f"  评估完成！汇总统计:")
    print(f"{'='*60}")
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if rows:
        psnr_vals = [float(r['PSNR_Out']) for r in rows]
        ssim_vals = [float(r['SSIM_Out']) for r in rows]
        bias_vals = [float(r['Bias_Out(%)']) for r in rows]

        print(f"  测试样本: {len(rows)}")
        print(f"  PSNR  (out):  {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB")
        print(f"  SSIM  (out):  {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")
        print(f"  Bias  (out):  {np.mean(bias_vals):.2f} ± {np.std(bias_vals):.2f} %")
    print(f"\n  可视化: {os.path.abspath(vis_dir)}")
    print(f"  ROI NPY: {os.path.abspath(npy_dir)}")
    print(f"  CSV:     {os.path.abspath(csv_path)}")


# ═══════════════════════════════════════════════════════════════
# 命令行参数 (已硬编码固定参数)
# ═══════════════════════════════════════════════════════════════

class DummyArgs:
    pass

if __name__ == '__main__':
    args = DummyArgs()

    # ── 强制硬编码你的模型参数 ──
    args.ckpt_path = './trido_output/checkpoint-199.pth'  # 200 epoch 模型
    args.data_path = 'I:/processed_data_trido/test'
    args.output_dir = './eval_results_trido'

    args.model_size = 'Large'
    args.use_sino_domain = True
    args.use_freq_domain = True
    args.img_size = 256
    args.patch_size = 16
    args.attn_dropout = 0.0
    args.proj_dropout = 0.0

    args.nfe = 50
    args.cfg_scale = 0.6
    args.P_mean = -0.5
    args.P_std = 1.2
    args.cond_drop_prob = 0.1

    args.batch_size = 64    # GPU 推理 batch size
    args.num_workers = 4    # DataLoader 线程数
    args.workers = 4        # 后台处理进程数

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evaluate_trido(args)
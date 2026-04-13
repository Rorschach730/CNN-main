import torch
import numpy as np
import os
import csv
import re
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
import queue

from denoiser_ud import Denoiser
from util.pet_dataset_ud import PETDenoisingDataset

from check_inference_ud import (
    InferenceConfig, get_mock_args, calculate_snr,
    robust_windowing, get_roi_bbox, calculate_suv_bias
)

# 强制开启 TensorFloat-32 (TF32) 进行极致矩阵加速 (专为 3090/A100 设计)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def background_worker(task_queue, result_queue, output_vis_dir, output_npy_dir):
    while True:
        try:
            task = task_queue.get(timeout=3)
            if task is None: break

            base_name, z_str, dose_val, gt_raw, in_raw, out_raw = task

            rmin, rmax, cmin, cmax = get_roi_bbox(gt_raw)
            if (rmax - rmin) <= 0 or (cmax - cmin) <= 0: continue

            roi_gt = gt_raw[rmin:rmax, cmin:cmax]
            roi_in = in_raw[rmin:rmax, cmin:cmax]
            roi_out = out_raw[rmin:rmax, cmin:cmax]

            dyn_range = roi_gt.max() - roi_gt.min()
            if dyn_range == 0: dyn_range = 1.0

            psnr_in = compare_psnr(roi_gt, roi_in, data_range=dyn_range)
            ssim_in = compare_ssim(roi_gt, roi_in, data_range=dyn_range)
            snr_in = calculate_snr(roi_gt, roi_in)
            mse_in = np.mean((roi_gt - roi_in) ** 2)
            bias_in = calculate_suv_bias(roi_gt, roi_in)

            psnr_out = compare_psnr(roi_gt, roi_out, data_range=dyn_range)
            ssim_out = compare_ssim(roi_gt, roi_out, data_range=dyn_range)
            snr_out = calculate_snr(roi_gt, roi_out)
            mse_out = np.mean((roi_gt - roi_out) ** 2)
            bias_out = calculate_suv_bias(roi_gt, roi_out)

            csv_row = [
                base_name, z_str, f"{dose_val:.3f}",
                f"{psnr_in:.4f}", f"{ssim_in:.4f}", f"{snr_in:.4f}", f"{mse_in:.6f}", f"{bias_in:.4f}",
                f"{psnr_out:.4f}", f"{ssim_out:.4f}", f"{snr_out:.4f}", f"{mse_out:.6f}", f"{bias_out:.4f}"
            ]

            result_queue.put(csv_row)

            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_gt.npy"), roi_gt)
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_in.npy"), roi_in)
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_out.npy"), roi_out)

            vmin, vmax = robust_windowing(gt_raw)
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            for ax, img, title in zip(axes, [in_raw, out_raw, gt_raw],
                                      [f"Input (Dose: {dose_val:.3f})", "JiT Denoised", "Ground Truth"]):
                ax.imshow(np.clip(img, 0, vmax), cmap='gray', vmin=vmin, vmax=vmax)
                ax.add_patch(
                    plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            fig.savefig(os.path.join(output_vis_dir, f"{base_name}.png"), dpi=100)
            plt.close(fig)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in worker: {e}")


def batch_evaluate_ud():
    mp.set_start_method('spawn', force=True)

    device = torch.device(InferenceConfig.device)

    test_data_folder = "./processed_data_udpet/test"
    output_vis_dir = "./test_visualizations_ud_test"
    output_npy_dir = "./test_npy_ud_test"
    output_csv_path = "test_visualizations_ud_test/ud_metrics_summary_test.csv"

    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print(f"--- 启动极速推理 (启用 TF32 与编译优化) ---")

    args = get_mock_args()
    model = Denoiser(args)
    model.to(device)

    ckpt = torch.load(InferenceConfig.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_ema', ckpt['model'])
    new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.net.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dataset = PETDenoisingDataset(test_data_folder, img_size=InferenceConfig.img_size)
    total_samples = len(dataset)
    if total_samples == 0: return

    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "Slice_Z", "Dose_Ratio",
            "PSNR_In", "SSIM_In", "SNR_In", "MSE_In", "Bias_In(%)",
            "PSNR_Out", "SSIM_Out", "SNR_Out", "MSE_Out", "Bias_Out(%)"
        ])

    # 【提速核心 2】：优化 Dataloader，防止 GPU 饿死
    # 如果 batch_size = 64 时显存才 12GB，我们可以直接把它拉到 128！
    eval_batch_size = 128

    # 将 worker 数量拉高，并开启持久化 worker，避免每个 epoch 重新创建进程的开销
    dataloader = DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    task_queue = mp.Queue(maxsize=200)
    result_queue = mp.Queue()

    # 后台只开 3 个进程足够了，多了反而抢 CPU
    num_workers = 3
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=background_worker, args=(task_queue, result_queue, output_vis_dir, output_npy_dir))
        p.start()
        workers.append(p)

    print(f"[*] 已启动后台进程，主进程开启狂飙模式... (Batch Size: {eval_batch_size})")

    print(f"[*] 已启动性能诊断探针，主进程开启... (Batch Size: {eval_batch_size})")

    import time
    sample_idx = 0

    t0 = time.time()
    for targets, conditions, doses in tqdm(dataloader, desc="GPU Inference"):
        t_data_load = time.time() - t0  # 记录读硬盘的时间

        B = targets.size(0)

        t1 = time.time()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs_t = model.generate(
                conditions.to(device, non_blocking=True), doses.to(device, non_blocking=True),
                steps=InferenceConfig.sampling_steps, cfg_scale=InferenceConfig.cfg_scale
            )
        # 等待 GPU 同步，确保计时准确
        torch.cuda.synchronize()
        t_gpu_compute = time.time() - t1  # 记录纯 GPU 矩阵乘法的时间

        t2 = time.time()
        out_raw_batch = outputs_t[:, 0, :, :].cpu().numpy()
        gt_raw_batch = targets[:, 0, :, :].cpu().numpy()
        in_raw_batch = conditions[:, 0, :, :].cpu().numpy()
        dose_batch = doses.view(-1).cpu().numpy()

        for i in range(B):
            file_path = dataset.samples[sample_idx]
            base_name = os.path.basename(file_path).replace('.pt', '')
            z_str = re.search(r'_Z(\d+)', base_name).group(1) if re.search(r'_Z(\d+)', base_name) else "0000"

            if gt_raw_batch[i].max() < 0.05:
                sample_idx += 1
                continue

            task_queue.put((
                base_name, z_str, dose_batch[i],
                gt_raw_batch[i], in_raw_batch[i], out_raw_batch[i]
            ))
            sample_idx += 1

        t_cpu_queue = time.time() - t2  # 记录数据转移和进程队列塞入时间

        # 打印性能诊断数据
        print(f"\n[探针] DataLoad: {t_data_load:.2f}s | GPU: {t_gpu_compute:.2f}s | CPU/Queue: {t_cpu_queue:.2f}s")

        batch_csv_rows = []
        while not result_queue.empty():
            batch_csv_rows.append(result_queue.get())
        if batch_csv_rows:
            with open(output_csv_path, mode='a', newline='') as f:
                csv.writer(f).writerows(batch_csv_rows)

        t0 = time.time()

    for _ in range(num_workers): task_queue.put(None)
    for p in workers: p.join()

    final_csv_rows = []
    while not result_queue.empty(): final_csv_rows.append(result_queue.get())
    if final_csv_rows:
        with open(output_csv_path, mode='a', newline='') as f:
            csv.writer(f).writerows(final_csv_rows)


if __name__ == "__main__":
    batch_evaluate_ud()
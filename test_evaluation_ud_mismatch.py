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
from check_inference_ud_mismatch import (
    InferenceConfig, get_mock_args, calculate_snr,
    robust_windowing, get_roi_bbox, calculate_suv_bias
)

# 强制开启 TensorFloat-32 (TF32) 进行极致矩阵加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def background_worker(task_queue, result_queue, output_vis_dir, output_npy_dir):
    while True:
        try:
            task = task_queue.get(timeout=3)
            if task is None: break

            # 接收真实剂量与推理标签
            base_name, z_str, true_dose, force_label, gt_raw, in_raw, out_raw = task

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

            # 判断匹配状态
            match_status = "Matched" if abs(true_dose - force_label) < 1e-4 else "Mismatched"

            csv_row = [
                base_name, z_str, f"{true_dose:.3f}", f"{force_label:.3f}", match_status,
                f"{psnr_in:.4f}", f"{ssim_in:.4f}", f"{snr_in:.4f}", f"{mse_in:.6f}", f"{bias_in:.4f}",
                f"{psnr_out:.4f}", f"{ssim_out:.4f}", f"{snr_out:.4f}", f"{mse_out:.6f}", f"{bias_out:.4f}"
            ]

            result_queue.put(csv_row)

            # 统一矩阵落盘
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_gt.npy"), roi_gt)
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_in.npy"), roi_in)
            np.save(os.path.join(output_npy_dir, f"{base_name}_roi_out.npy"), roi_out)

            # 绘图逻辑
            vmin, vmax = robust_windowing(gt_raw)
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))

            title_in = f"Input (Physical: {true_dose:.3f})"
            title_out = f"Denoised (Label: {force_label:.3f} | {match_status})"
            title_gt = "Ground Truth"

            for ax, img, title in zip(axes, [in_raw, out_raw, gt_raw], [title_in, title_out, title_gt]):
                ax.imshow(np.clip(img, 0, vmax), cmap='gray', vmin=vmin, vmax=vmax)
                ax.add_patch(
                    plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
                ax.set_title(title, fontsize=11)
                ax.axis('off')

            plt.tight_layout()
            fig.savefig(os.path.join(output_vis_dir, f"{base_name}.png"), dpi=100)
            plt.close(fig)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in worker: {e}")


def batch_evaluate_ud(force_label):
    device = torch.device(InferenceConfig.device)

    # 根据当前推理标签隔离输出路径
    suffix = f"_Label_{force_label:.3f}"
    test_data_folder = "./processed_data_udpet/test"
    output_vis_dir = f"./test_visualizations_ud_test{suffix}"
    output_npy_dir = f"./test_npy_ud_test{suffix}"
    output_csv_path = f"{output_vis_dir}/ud_metrics_summary_test{suffix}.csv"

    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print(f"\n--- 启动批量推理 [强制使用标签: {force_label:.3f}] ---")

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
            "Sample_ID", "Slice_Z", "Physical_Dose", "Inference_Label", "Match_Status",
            "PSNR_In", "SSIM_In", "SNR_In", "MSE_In", "Bias_In(%)",
            "PSNR_Out", "SSIM_Out", "SNR_Out", "MSE_Out", "Bias_Out(%)"
        ])

    eval_batch_size = 128
    dataloader = DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    task_queue = mp.Queue(maxsize=200)
    result_queue = mp.Queue()

    num_workers = 3
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=background_worker, args=(task_queue, result_queue, output_vis_dir, output_npy_dir))
        p.start()
        workers.append(p)

    print(f"[*] 已启动后台进程，主进程开启狂飙模式... (Batch Size: {eval_batch_size})")

    import time
    sample_idx = 0

    t0 = time.time()
    for targets, conditions, true_doses in tqdm(dataloader, desc=f"Inference {suffix}"):
        t_data_load = time.time() - t0

        B = targets.size(0)

        # 强制覆写推理标签
        inference_doses = torch.full_like(true_doses, force_label)

        t1 = time.time()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs_t = model.generate(
                conditions.to(device, non_blocking=True), inference_doses.to(device, non_blocking=True),
                steps=InferenceConfig.sampling_steps, cfg_scale=InferenceConfig.cfg_scale
            )
        torch.cuda.synchronize()
        t_gpu_compute = time.time() - t1

        t2 = time.time()
        out_raw_batch = outputs_t[:, 0, :, :].cpu().numpy()
        gt_raw_batch = targets[:, 0, :, :].cpu().numpy()
        in_raw_batch = conditions[:, 0, :, :].cpu().numpy()

        # 保留真实的物理剂量值用于记录
        true_dose_batch = true_doses.view(-1).cpu().numpy()

        for i in range(B):
            file_path = dataset.samples[sample_idx]
            base_name = os.path.basename(file_path).replace('.pt', '')
            z_str = re.search(r'_Z(\d+)', base_name).group(1) if re.search(r'_Z(\d+)', base_name) else "0000"

            if gt_raw_batch[i].max() < 0.05:
                sample_idx += 1
                continue

            # 传入真实剂量和使用的标签
            task_queue.put((
                base_name, z_str, true_dose_batch[i], force_label,
                gt_raw_batch[i], in_raw_batch[i], out_raw_batch[i]
            ))
            sample_idx += 1

        t_cpu_queue = time.time() - t2

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

    # 释放显存，为下一次不同标签的循环做准备
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # 全量测试标签池：依次作为模型的推理条件
    test_labels = [0.1, 0.25, 0.5]

    for label in test_labels:
        batch_evaluate_ud(force_label=label)

    print("\n[+] 所有剂量标签配置的批量评估任务均已完成。")
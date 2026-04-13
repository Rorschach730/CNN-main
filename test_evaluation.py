import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from denoiser_new import Denoiser
from util.pet_dataset_new import PETDenoisingDataset

# 确保从此模块导入新增的 calculate_clinical_metrics 和 get_roi_bbox
from check_inference_new import (
    InferenceConfig,
    get_mock_args,
    calculate_snr,
    robust_windowing,
    get_roi_bbox,
    calculate_clinical_metrics
)


def batch_evaluate():
    # 强制关闭 matplotlib 的交互模式，防止批量绘图时内存泄漏与渲染阻塞
    plt.ioff()

    device = torch.device(InferenceConfig.device)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    # 将路径指向 test 集
    # 将路径指向 test 集
    test_data_folder = "./processed_data_3d_osem/test"
    output_vis_dir = "./test_visualizations"
    output_csv_path = "./test_visualizations/test_metrics_summary.csv"

    os.makedirs(output_vis_dir, exist_ok=True)

    print(f"--- 启动大批量测试集推理与 ROI 量化落盘 (临床指标 + CV指标) ---")
    print(f"[*] 目标数据路径: {test_data_folder}")

    args = get_mock_args()
    model = Denoiser(args)
    model.to(device)

    print(f"[*] 正在加载权重: {InferenceConfig.checkpoint_path}")
    try:
        ckpt = torch.load(InferenceConfig.checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[!] 致命错误：找不到权重文件 {InferenceConfig.checkpoint_path}。")
        return

    state_dict = ckpt.get('model_ema', ckpt['model'])
    new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.net.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dataset = PETDenoisingDataset(test_data_folder, img_size=InferenceConfig.img_size)
    total_samples = len(dataset)
    if total_samples == 0:
        print("[!] 致命错误：测试集为空，请检查路径。")
        return

    print(f"[*] 共检测到 {total_samples} 个测试样本，开始推理并生成报告...")

    # 初始化 CSV 统计表头 (包含全维度指标)
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "File_Name", "Slice_Z",
            "ROI_PSNR_In", "ROI_SSIM_In", "ROI_SNR_In", "ROI_Bias_In(%)", "ROI_MAPE_In(%)", "ROI_CR_In(%)",
            "ROI_PSNR_Out", "ROI_SSIM_Out", "ROI_SNR_Out", "ROI_Bias_Out(%)", "ROI_MAPE_Out(%)", "ROI_CR_Out(%)"
        ])

    for idx in tqdm(range(total_samples), desc="Testing & Saving"):
        file_path, z_slice = dataset.samples[idx]
        base_name = os.path.basename(file_path).replace('.npy', '')
        sample_id = f"{base_name}_Z{z_slice:03d}"

        target, condition = dataset[idx]

        # 过滤极端空张量
        if target.max() < 0.05:
            continue

        target_t = target.unsqueeze(0).to(device)
        condition_t = condition.unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model.generate(condition_t, steps=InferenceConfig.sampling_steps,
                                           cfg_scale=InferenceConfig.cfg_scale)

        # 1. 提取未经截断的纯正物理张量 (Raw)
        out_raw = output_tensor.squeeze().cpu().numpy()
        gt_raw = target_t.squeeze().cpu().numpy()
        in_raw = condition_t.squeeze().cpu().numpy()

        # 2. 提取截断张量 (01)
        out_01 = np.clip(out_raw, 0.0, 1.0)
        gt_01 = np.clip(gt_raw, 0.0, 1.0)
        in_01 = np.clip(in_raw, 0.0, 1.0)

        # 3. 生成 ROI 包围盒
        rmin, rmax, cmin, cmax = get_roi_bbox(gt_01)

        if (rmax - rmin) <= 0 or (cmax - cmin) <= 0:
            continue

        # --- ROI 切片分配 ---
        # [物理修正]：必须使用 raw 数据进入临床指标运算
        roi_gt_raw = gt_raw[rmin:rmax, cmin:cmax]
        roi_in_raw = in_raw[rmin:rmax, cmin:cmax]
        roi_out_raw = out_raw[rmin:rmax, cmin:cmax]

        # CV 指标仍沿用 01 截断数据
        roi_gt_01 = gt_01[rmin:rmax, cmin:cmax]
        roi_in_01 = in_01[rmin:rmax, cmin:cmax]
        roi_out_01 = out_01[rmin:rmax, cmin:cmax]

        # --- 指标运算 ---
        # 临床核医学指标
        bias_in, mape_in, cr_in = calculate_clinical_metrics(roi_gt_raw, roi_in_raw)
        bias_out, mape_out, cr_out = calculate_clinical_metrics(roi_gt_raw, roi_out_raw)

        # CV 通用指标
        psnr_in = compare_psnr(roi_gt_01, roi_in_01, data_range=1.0)
        ssim_in = compare_ssim(roi_gt_01, roi_in_01, data_range=1.0)
        snr_in = calculate_snr(roi_gt_01, roi_in_01)

        psnr_out = compare_psnr(roi_gt_01, roi_out_01, data_range=1.0)
        ssim_out = compare_ssim(roi_gt_01, roi_out_01, data_range=1.0)
        snr_out = calculate_snr(roi_gt_01, roi_out_01)

        # 写入 CSV
        with open(output_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                sample_id, base_name, z_slice,
                f"{psnr_in:.4f}", f"{ssim_in:.4f}", f"{snr_in:.4f}", f"{bias_in:.4f}", f"{mape_in:.4f}", f"{cr_in:.4f}",
                f"{psnr_out:.4f}", f"{ssim_out:.4f}", f"{snr_out:.4f}", f"{bias_out:.4f}", f"{mape_out:.4f}",
                f"{cr_out:.4f}"
            ])

        # --- 可视化与落盘 ---
        str_in = f"Noisy (ROI)\nBias: {bias_in:.1f}% | MAPE: {mape_in:.1f}% | CR: {cr_in:.1f}%\nPSNR: {psnr_in:.2f} | SSIM: {ssim_in:.3f}"
        str_out = f"JiT-Denoised (ROI)\nBias: {bias_out:.1f}% | MAPE: {mape_out:.1f}% | CR: {cr_out:.1f}%\nPSNR: {psnr_out:.2f} | SSIM: {ssim_out:.3f}"

        vmin, vmax = robust_windowing(gt_01)
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        axes[0].imshow(in_01, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].add_patch(
            plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
        axes[0].set_title(str_in, fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(out_01, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].add_patch(
            plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
        axes[1].set_title(str_out, fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(gt_01, cmap='gray', vmin=vmin, vmax=vmax)
        axes[2].add_patch(
            plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False, edgecolor='red', linewidth=1))
        axes[2].set_title("Ground Truth (Clean)", fontsize=10)
        axes[2].axis('off')

        plt.tight_layout()
        save_file = os.path.join(output_vis_dir, f"{sample_id}.png")
        plt.savefig(save_file, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    print(f"\n[完成] 测试集 ROI 临床与CV指标量化处理完毕。")
    print(f"- 可视化图像目录: {output_vis_dir}")
    print(f"- 统计分析表路径: {output_csv_path}")


if __name__ == "__main__":
    batch_evaluate()
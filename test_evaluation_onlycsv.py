import numpy as np
import os
import csv
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 导入你原有的数据集类和指标函数（路径与 test_evaluation.py 中一致）
from util.pet_dataset_new import PETDenoisingDataset
from check_inference_new import (
    InferenceConfig,
    calculate_snr,
    get_roi_bbox,
    calculate_clinical_metrics
)


def fast_evaluate_input():
    """
    仅遍历测试集，计算噪声输入与干净参考的 ROI 指标并存入 CSV。
    不进行模型推理，不生成图片。
    """
    test_data_folder = "./processed_data_3d_osem/test"
    output_csv_path = "./test_input_metrics.csv"   # 新文件名，避免覆盖原有完整报告

    print(f"--- 快速评估噪声输入与 Ground Truth 的 ROI 指标 ---")
    print(f"[*] 目标数据路径: {test_data_folder}")

    dataset = PETDenoisingDataset(test_data_folder, img_size=InferenceConfig.img_size)
    total_samples = len(dataset)
    if total_samples == 0:
        print("[!] 致命错误：测试集为空，请检查路径。")
        return

    print(f"[*] 共检测到 {total_samples} 个 2D 切片样本，开始计算指标...")

    # 准备 CSV 文件头（仅 Input 指标）
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "File_Name", "Slice_Z",
            "ROI_PSNR_In", "ROI_SSIM_In", "ROI_SNR_In",
            "ROI_Bias_In(%)", "ROI_MAPE_In(%)", "ROI_CR_In(%)"
        ])

    for idx in tqdm(range(total_samples), desc="Evaluating Input"):
        file_path, z_slice = dataset.samples[idx]
        base_name = os.path.basename(file_path).replace('.npy', '')
        sample_id = f"{base_name}_Z{z_slice:03d}"

        target, condition = dataset[idx]

        # 过滤掉几乎全黑的切片（保留原逻辑）
        if target.max() < 0.05:
            continue

        # 转为 numpy 数组
        gt_raw = target.squeeze().numpy()
        in_raw = condition.squeeze().numpy()

        # 生成 ROI 边界框（基于 GT 的 01 截断版本）
        gt_01 = np.clip(gt_raw, 0.0, 1.0)
        rmin, rmax, cmin, cmax = get_roi_bbox(gt_01)
        if (rmax - rmin) <= 0 or (cmax - cmin) <= 0:
            continue

        # 提取 ROI 区域（临床指标使用 raw 数据）
        roi_gt_raw = gt_raw[rmin:rmax, cmin:cmax]
        roi_in_raw = in_raw[rmin:rmax, cmin:cmax]

        # 提取 ROI 区域（CV 指标使用 01 截断数据）
        roi_gt_01 = gt_01[rmin:rmax, cmin:cmax]
        roi_in_01 = np.clip(in_raw[rmin:rmax, cmin:cmax], 0.0, 1.0)

        # 计算临床指标
        bias_in, mape_in, cr_in = calculate_clinical_metrics(roi_gt_raw, roi_in_raw)

        # 计算 CV 通用指标
        psnr_in = compare_psnr(roi_gt_01, roi_in_01, data_range=1.0)
        ssim_in = compare_ssim(roi_gt_01, roi_in_01, data_range=1.0)
        snr_in = calculate_snr(roi_gt_01, roi_in_01)

        # 写入 CSV
        with open(output_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                sample_id, base_name, z_slice,
                f"{psnr_in:.4f}", f"{ssim_in:.4f}", f"{snr_in:.4f}",
                f"{bias_in:.4f}", f"{mape_in:.4f}", f"{cr_in:.4f}"
            ])

    print(f"\n[完成] 输入噪声图 ROI 指标计算完毕。")
    print(f"- 结果已保存至: {output_csv_path}")


if __name__ == "__main__":
    fast_evaluate_input()
import torch
import numpy as np
import os
import csv
import re
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu

# 导入 UDPET 专用的模型与数据管道
from denoiser_ud import Denoiser
from util.pet_dataset_ud import PETDenoisingDataset
from check_inference_ud import InferenceConfig, get_mock_args, get_roi_bbox


def calculate_clinical_metrics_otsu(roi_gt, roi_pred):
    """
    基于动态大津法(Otsu)的无掩膜临床定量计算
    输入必须是 Raw Tensor (未经过 clip 的绝对物理 SUV 域)
    """
    gt_max = np.max(roi_gt)
    pred_max = np.max(roi_pred)
    eps = 1e-8

    # 1. SUVmax Bias (%) - 严格保留正负号，评估高估或低估倾向
    bias_max = (pred_max - gt_max) / (gt_max + eps) * 100.0

    # 2. MAPE (%) - 仅评估代谢活跃区 (>10% SUVmax)，必须加绝对值
    active_mask = roi_gt > (gt_max * 0.1)
    if np.sum(active_mask) == 0:
        active_mask = roi_gt > 0  # 极端平滑区域退守策略

    mape = np.mean(np.abs(roi_pred[active_mask] - roi_gt[active_mask]) / (roi_gt[active_mask] + eps)) * 100.0

    # 3. Contrast Recovery (CR) (%) - 大津法动态阈值双峰分割
    try:
        # 使用 Otsu 动态寻找当前 ROI 内病灶与背景的最佳阈值
        otsu_thresh = threshold_otsu(roi_gt)
        lesion_mask = roi_gt >= otsu_thresh
        bg_mask = roi_gt < otsu_thresh

        if np.sum(lesion_mask) > 0 and np.sum(bg_mask) > 0:
            c_gt = np.mean(roi_gt[lesion_mask]) / (np.mean(roi_gt[bg_mask]) + eps)
            c_pred = np.mean(roi_pred[lesion_mask]) / (np.mean(roi_pred[bg_mask]) + eps)
            cr = (c_pred / (c_gt + eps)) * 100.0
        else:
            cr = 100.0
    except Exception:
        # 当 ROI 内部像素完全一致（方差为0），Otsu 会报错，直接返回 100%
        cr = 100.0

    return bias_max, mape, cr


def batch_evaluate_clinical():
    device = torch.device(InferenceConfig.device)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    test_data_folder = "./processed_data_udpet/test"
    output_csv_path = "./ud_clinical_metrics.csv"

    # 若之前存在残留文件，先清理
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    print(f"--- 启动 UDPET 纯临床定量指标 (SUVmax Bias / MAPE / Otsu-CR) 极速评估 ---")

    args = get_mock_args()
    model = Denoiser(args).to(device)

    print(f"[*] 加载权重: {InferenceConfig.checkpoint_path}")
    ckpt = torch.load(InferenceConfig.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_ema', ckpt['model'])
    new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.net.load_state_dict(new_state_dict, strict=False)
    model.eval()

    dataset = PETDenoisingDataset(test_data_folder, img_size=InferenceConfig.img_size)
    total_samples = len(dataset)
    if total_samples == 0:
        print("[!] 测试集为空。")
        return

    # 写入全新表头
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_ID", "Dose_Ratio",
            "SUVmax_Bias_In(%)", "MAPE_In(%)", "CR_In(%)",
            "SUVmax_Bias_Out(%)", "MAPE_Out(%)", "CR_Out(%)"
        ])

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    sample_idx = 0
    for targets, conditions, doses in tqdm(dataloader, desc="Evaluating Clinical Metrics"):
        B = targets.size(0)

        targets_t = targets.to(device)
        conditions_t = conditions.to(device)
        doses_t = doses.to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs_t = model.generate(
                conditions_t, doses_t,
                steps=InferenceConfig.sampling_steps,
                cfg_scale=InferenceConfig.cfg_scale
            )

        out_raw_batch = outputs_t[:, 0, :, :].cpu().numpy()
        gt_raw_batch = targets_t[:, 0, :, :].cpu().numpy()
        in_raw_batch = conditions_t[:, 0, :, :].cpu().numpy()
        dose_batch = doses_t.view(-1).cpu().numpy()

        batch_csv_rows = []

        for i in range(B):
            file_path = dataset.samples[sample_idx]
            base_name = os.path.basename(file_path).replace('.pt', '')

            gt_raw = gt_raw_batch[i]
            in_raw = in_raw_batch[i]
            out_raw = out_raw_batch[i]
            dose_val = dose_batch[i]

            # 过滤全黑无效背景层
            if gt_raw.max() < 0.05:
                sample_idx += 1
                continue

            # 定位边界框（寻址依然使用安全的 0-1 截断域，防止飞点干扰 BBox）
            gt_01 = np.clip(gt_raw, 0.0, 1.0)
            rmin, rmax, cmin, cmax = get_roi_bbox(gt_01)
            if (rmax - rmin) <= 0 or (cmax - cmin) <= 0:
                sample_idx += 1
                continue

            # 提取 Raw Tensor ROI（严禁截断）
            roi_gt_raw = gt_raw[rmin:rmax, cmin:cmax]
            roi_in_raw = in_raw[rmin:rmax, cmin:cmax]
            roi_out_raw = out_raw[rmin:rmax, cmin:cmax]

            # 执行临床计算
            bias_in, mape_in, cr_in = calculate_clinical_metrics_otsu(roi_gt_raw, roi_in_raw)
            bias_out, mape_out, cr_out = calculate_clinical_metrics_otsu(roi_gt_raw, roi_out_raw)

            batch_csv_rows.append([
                base_name, f"{dose_val:.3f}",
                f"{bias_in:.4f}", f"{mape_in:.4f}", f"{cr_in:.4f}",
                f"{bias_out:.4f}", f"{mape_out:.4f}", f"{cr_out:.4f}"
            ])
            sample_idx += 1

        if batch_csv_rows:
            with open(output_csv_path, mode='a', newline='') as f:
                csv.writer(f).writerows(batch_csv_rows)


if __name__ == "__main__":
    batch_evaluate_clinical()
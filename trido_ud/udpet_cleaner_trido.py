"""
UDPET Cleaner (TriDo v2) — Optimized Data Pipeline
====================================================
改进点:
  1. float16 存储 (源 DICOM 为 int16, float16 无损覆盖 → 2x 缩减)
  2. 保留脑部+躯干切片 (0–670mm from top, 去掉下肢 → ~40% 缩减)
  3. train/val/test = 7:1.5:1.5 病人级切分 (防数据泄漏)
  4. 按 manufacturer/gender/age 分层抽样
  5. torch.save 内置 zip 压缩 (额外 ~10-15%)

预计: 210 GB → ~60 GB

输出结构:
  processed_data_trido/
  ├── train/P0001/P0001_D10_Z0050.pt ...
  ├── val/  P0100/...
  └── test/ P0200/...

兼容: pet_dataset_trido.py 直接读取, 格式不变 [2, H, W] float16
"""

import os
import re
import random
import pydicom
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


class UDPETCleanerConfig:
    ROOT_DIRS = [
        "H:/Bern-Inselspital-2022",
        "H:/Shanghai-Ruijin-Hospital-2022",
        "H:/Shanghai-Ruijin-Hospital-2023",
    ]
    OUTPUT_DIR = "I:/processed_data_trido"
    TARGET_SIZE = 256

    # ── 过滤窗口 (mm from top): 0=脑顶, 保留脑部+躯干, 去掉下肢 ──
    TORSO_START_MM = 0.0
    TORSO_LENGTH_MM = 670.0  # 窗口宽度: 0–670mm = 脑+胸+腹

    # ── train/val/test 比例 ──
    SPLIT_RATIOS = (0.70, 0.15, 0.15)
    SEED = 42

    # ── 剂量关键词映射 ──
    DOSE_MAPPING = {
        "1-2": 2, "d2": 2, "1_2": 2,
        "1-4": 4, "d4": 4, "1_4": 4,
        "1-10": 10, "d10": 10, "1_10": 10,
    }

    # ── 保存精度: float16 = 2 bytes, float32 = 4 bytes ──
    SAVE_DTYPE = torch.float16


# ═══════════════════════════════════════════════════════════════════════
#  SUV 计算
# ═══════════════════════════════════════════════════════════════════════

def calculate_suv_factor(dcm_hdr):
    """从 DICOM 头计算 SUV 转换因子。失败返回 None。"""
    try:
        weight = float(dcm_hdr.PatientWeight) * 1000.0  # kg → g
        rad_seq = dcm_hdr.RadiopharmaceuticalInformationSequence[0]
        dose = float(rad_seq.RadionuclideTotalDose)
        slope = float(getattr(dcm_hdr, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm_hdr, "RescaleIntercept", 0.0))
        return (slope * weight) / dose, intercept
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════
#  图像处理
# ═══════════════════════════════════════════════════════════════════════

def center_crop_numpy(img_array, target_size=256):
    """中心裁剪 + 小图零填充, 返回 (target_size, target_size) float32."""
    h, w = img_array.shape
    th, tw = target_size, target_size
    if h < th or w < tw:
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        img_array = np.pad(
            img_array,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="constant",
        )
        h, w = img_array.shape
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return img_array[i : i + th, j : j + tw]


# ═══════════════════════════════════════════════════════════════════════
#  病人发现与剂量解析
# ═══════════════════════════════════════════════════════════════════════

def _is_full_dose_dir(dirpath):
    """Check if a directory is a full-dose scan directory by its name.

    Strategy A (Bern-Inselspital): basename contains 'full dose', 'drf_100', '100 dose'
    Strategy B (Shanghai-Ruijin):  basename contains 'normal' (WB scan protocol)
    """
    basename_lower = os.path.basename(dirpath).lower()
    # Bern pattern
    if "full dose" in basename_lower or "100 dose" in basename_lower or "drf_100" in basename_lower:
        return True
    # Shanghai pattern: "2.886 x 600 WB NORMAL"
    if "normal" in basename_lower:
        return True
    return False


def _is_dicom_dir(dirpath):
    """Check if a directory contains at least one DICOM file."""
    try:
        for f in os.listdir(dirpath):
            if f.lower().endswith((".dcm", ".ima")):
                return True
    except Exception:
        pass
    return False


def _parse_wb_param(dirname):
    """Extract the numeric parameter from a WB scan protocol directory name.

    Example: '2.886 x 600 WB NORMAL' → 600, '2.886 x 150 WB NORMAL' → 150
    Returns None if the name doesn't match the WB pattern.
    """
    m = re.match(r"[\d.]+\s*x\s*(\d+)\s*wb", dirname.lower().strip())
    if m:
        return int(m.group(1))
    return None


def _resolve_low_dose_denom(low_dirname, full_dirname, full_param):
    """Determine dose denominator for a low-dose directory.

    Strategy A (Bern): DOSE_MAPPING keywords in directory name → explicit denom
    Strategy B (Shanghai): WB protocol name → ratio of full_param / low_param
    Returns (denom, is_match) — is_match=False means skip this directory.
    """
    dl = low_dirname.lower().strip()

    # Strategy A: Bern-style dose keywords (1-2, d4, 1_10, etc.)
    for keyword, denom in UDPETCleanerConfig.DOSE_MAPPING.items():
        if keyword in dl and "1000" not in dl:
            return denom, True

    # Strategy B: Shanghai-style WB protocol → ratio
    if full_param is not None:
        low_param = _parse_wb_param(low_dirname)
        if low_param is not None and low_param > 0:
            if full_param > low_param and full_param % low_param == 0:
                denom = full_param // low_param
                if denom > 1:
                    return denom, True

    # Strategy C: universal fallback — directory contains DICOM but no dose info
    # Return denom=-1 as a flag meaning "unknown dose, try to infer later"
    return -1, False


def discover_patients(root_dirs):
    """
    扫描所有数据源, 构建病人清单。

    采用「自上而下」策略:
      1. Walk 整棵树, 找到所有 full-dose 目录 (名称含 full/normal/drf_100)
      2. 从父目录(病人级)发现所有低剂量兄弟目录
      3. 多策略解析剂量分母 (Bern 关键词 + Shanghai WB比值 + 通用回退)

    返回: list of dict {
        patient_dir, patient_id, full_dose_dir,
        low_dose_pairs: [(path, dose_denom), ...],
        uid, gender, age, manufacturer
    }
    """
    patients = []
    seen_full_dose = set()  # 防止同一个 full_dose_dir 被重复处理

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"  [跳过] 路径不存在: {root_dir}")
            continue

        print(f"  [扫描] {root_dir} ...")
        for dirpath, dirnames, _ in os.walk(root_dir):
            # 只关注 full-dose 目录本身 (basename 含 full/normal/drf_100)
            if not _is_full_dose_dir(dirpath):
                continue

            full_dose_dir = dirpath
            if full_dose_dir in seen_full_dose:
                continue
            seen_full_dose.add(full_dose_dir)

            # 父目录 = 病人/采集目录
            patient_dir = os.path.dirname(full_dose_dir)
            if not os.path.isdir(patient_dir):
                continue

            # 解析 full-dose 的 WB 参数 (Shanghai), 用于后续低剂量比值计算
            full_wb_param = _parse_wb_param(os.path.basename(full_dose_dir))

            # ── 发现所有兄弟目录作为候选剂量目录 ──
            try:
                siblings = os.listdir(patient_dir)
            except Exception:
                continue

            low_dose_pairs = []
            for sib in siblings:
                sib_path = os.path.join(patient_dir, sib)
                if not os.path.isdir(sib_path) or sib_path == full_dose_dir:
                    continue

                denom, is_low_dose = _resolve_low_dose_denom(
                    sib, os.path.basename(full_dose_dir), full_wb_param
                )
                if not is_low_dose:
                    continue
                low_dose_pairs.append((sib_path, denom))

            # ── 额外扫描: Bern 数据中可能有 dose 子目录嵌套在日期文件夹下,
            #     但有些低剂量文件夹未直接匹配关键词 (如 'dose_1-2' vs '1-2_dose')
            #     对任何含 DICOM 文件的兄弟目录且不在现有列表中的, 再尝试匹配 ──
            for sib in siblings:
                sib_path = os.path.join(patient_dir, sib)
                if (not os.path.isdir(sib_path) or sib_path == full_dose_dir
                        or any(sib_path == lp[0] for lp in low_dose_pairs)):
                    continue
                if _is_dicom_dir(sib_path):
                    # 尝试 Bern 关键词
                    denom, matched = _resolve_low_dose_denom(
                        sib, os.path.basename(full_dose_dir), full_wb_param
                    )
                    if matched:
                        low_dose_pairs.append((sib_path, denom))

            if not low_dose_pairs:
                continue

            # ── 读 DICOM 元数据 ──
            try:
                dcm_files = [
                    f for f in os.listdir(full_dose_dir)
                    if f.lower().endswith((".dcm", ".ima"))
                ]
                if not dcm_files:
                    continue
                dcm = pydicom.dcmread(
                    os.path.join(full_dose_dir, dcm_files[0]),
                    stop_before_pixels=True,
                )
                uid = getattr(dcm, "StudyInstanceUID", patient_dir)
                gender = getattr(dcm, "PatientSex", "Unknown")
                age_raw = getattr(dcm, "PatientAge", "000Y")
                age = f"{age_raw[:2]}0s" if len(age_raw) >= 2 else "Unknown"
                manufacturer = getattr(dcm, "Manufacturer", "Unknown")

                # FDG tracer 检查
                tracer = "Unknown"
                if "RadiopharmaceuticalInformationSequence" in dcm:
                    tracer = getattr(
                        dcm.RadiopharmaceuticalInformationSequence[0],
                        "Radiopharmaceutical",
                        "Unknown",
                    )
                if "FDG" not in tracer.upper() and "FLUORODEOXYGLUCOSE" not in tracer.upper():
                    continue

            except Exception:
                continue

            # 使用父目录 basename 作为可读 ID (Bern: 日期文件夹, Shanghai: Anonymous_ANO_xxxx)
            patient_id = os.path.basename(patient_dir)

            patients.append(
                {
                    "patient_dir": patient_dir,
                    "patient_id": patient_id,
                    "full_dose_dir": full_dose_dir,
                    "low_dose_pairs": low_dose_pairs,
                    "uid": uid,
                    "gender": gender,
                    "age": age,
                    "manufacturer": manufacturer,
                }
            )

    return patients


# ═══════════════════════════════════════════════════════════════════════
#  病人级分层抽样 (7:1.5:1.5)
# ═══════════════════════════════════════════════════════════════════════

def stratified_split_by_patient(patients, ratios, seed):
    """
    按 manufacturer × gender × age 分层, 以病人(uid)为单位切分。
    同一病人的所有剂量对始终在同一个集合中。
    """
    random.seed(seed)
    np.random.seed(seed)

    # 按 uid 去重 (同一病人只计一次)
    seen_uids = set()
    unique_patients = []
    for p in patients:
        if p["uid"] not in seen_uids:
            seen_uids.add(p["uid"])
            unique_patients.append(p)

    # 分层分组
    strata = defaultdict(list)
    for p in unique_patients:
        key = (p["manufacturer"], p["gender"], p["age"])
        strata[key].append(p)

    train_uids, val_uids, test_uids = set(), set(), set()

    for key, group in strata.items():
        n = len(group)
        # 按 uid 排序以保持确定性
        group.sort(key=lambda x: x["uid"])
        indices = list(range(n))
        np.random.shuffle(indices)

        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1])))
        n_test = max(1, n - n_train - n_val)

        # 确保至少各 1 个 (如果组够大)
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:  # n == 1
            n_train, n_val, n_test = 1, 0, 0

        train_uids.update(p["uid"] for p in [group[i] for i in indices[:n_train]])
        val_uids.update(
            p["uid"] for p in [group[i] for i in indices[n_train : n_train + n_val]]
        )
        test_uids.update(
            p["uid"] for p in [group[i] for i in indices[n_train + n_val :]]
        )

    return train_uids, val_uids, test_uids


# ═══════════════════════════════════════════════════════════════════════
#  主处理逻辑
# ═══════════════════════════════════════════════════════════════════════

def process_patient(patient, output_base, split_name, config):
    """
    处理单个病人的一组 (full-dose, low-dose) 配对。
    仅保存 0–670mm 范围 (脑部+躯干, 去掉下肢)。
    输出 float16 .pt 文件。
    """
    full_dir = patient["full_dose_dir"]
    target_size = config.TARGET_SIZE
    torso_start = config.TORSO_START_MM
    torso_end = config.TORSO_START_MM + config.TORSO_LENGTH_MM
    save_dtype = config.SAVE_DTYPE

    # ── 加载 full-dose slices ──
    full_slices = []
    for f in os.listdir(full_dir):
        if not (f.lower().endswith(".dcm") or f.lower().endswith(".ima")):
            continue
        try:
            dcm = pydicom.dcmread(os.path.join(full_dir, f))
            factor, intercept = calculate_suv_factor(dcm)
            if factor is None:
                continue
            z_pos = float(dcm.ImagePositionPatient[2])
            pixel_array = dcm.pixel_array.astype(np.float32)
            suv_array = pixel_array * factor + intercept
            full_slices.append((z_pos, suv_array))
        except Exception:
            continue

    if not full_slices:
        return 0

    # ── 过滤: 仅保留 0–670mm 范围 (脑部+躯干, 去掉下肢) ──
    # DICOM Z coordinates in this dataset: Z↑ = toward feet (Z小=头部, Z大=脚部).
    # Sort ascending → head first, feet last.
    full_slices.sort(key=lambda x: x[0])
    head_z = full_slices[0][0]  # smallest Z = top of head
    torso_full = []
    for fz, fpx in full_slices:
        dist_from_head = fz - head_z  # always >= 0 in ascending order
        if torso_start <= dist_from_head <= torso_end:
            torso_full.append((fz, fpx))

    if not torso_full:
        return 0

    output_count = 0

    # ── 遍历每种低剂量 ──
    for low_dir, dose_denom in patient["low_dose_pairs"]:
        # 加载低剂量切片
        low_slices = {}
        for f in os.listdir(low_dir):
            if not (f.lower().endswith(".dcm") or f.lower().endswith(".ima")):
                continue
            try:
                dcm = pydicom.dcmread(os.path.join(low_dir, f))
                factor, intercept = calculate_suv_factor(dcm)
                if factor is None:
                    continue
                z_pos = float(dcm.ImagePositionPatient[2])
                suv_array = (
                    dcm.pixel_array.astype(np.float32) * factor + intercept
                )
                low_slices[z_pos] = center_crop_numpy(suv_array, target_size)
            except Exception:
                continue

        if not low_slices:
            continue

        # ── Z 对齐 + 保存 ──
        save_dir = os.path.join(output_base, split_name, patient["patient_id"])
        os.makedirs(save_dir, exist_ok=True)

        for i, (fz, fpx) in enumerate(torso_full):
            # 找最接近的 Z 坐标 (容差 0.5mm)
            matched_z = min(
                (lz for lz in low_slices if abs(lz - fz) < 0.5),
                key=lambda lz: abs(lz - fz),
                default=None,
            )
            if matched_z is None:
                continue

            target_crop = center_crop_numpy(fpx, target_size)
            cond_crop = low_slices[matched_z]

            # [2, H, W] float16
            tensor_pair = torch.stack(
                [
                    torch.from_numpy(cond_crop),
                    torch.from_numpy(target_crop),
                ]
            ).to(save_dtype)

            save_name = f"{patient['patient_id']}_D{dose_denom}_Z{i:04d}.pt"
            save_path = os.path.join(save_dir, save_name)

            # torch.save 默认 zip 序列化 (PyTorch ≥1.6)
            torch.save(tensor_pair, save_path, _use_new_zipfile_serialization=True)
            output_count += 1

    return output_count


# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    config = UDPETCleanerConfig()

    # ── 阶段 1: 扫描所有病人 ──
    print("=" * 60)
    print("[阶段 1] 扫描数据源, 构建病人清单...")
    print("=" * 60)
    patients = discover_patients(config.ROOT_DIRS)

    if not patients:
        print("[错误] 未发现任何有效病人数据, 退出。")
        return

    unique_uids = len(set(p["uid"] for p in patients))
    total_pairs = sum(len(p["low_dose_pairs"]) for p in patients)
    print(f"  → 发现 {unique_uids} 位病人, 共 {total_pairs} 组剂量配对")

    # ── 阶段 2: 病人级分层切分 ──
    print("\n" + "=" * 60)
    print("[阶段 2] 病人级分层抽样 (7:1.5:1.5)...")
    print("=" * 60)
    train_uids, val_uids, test_uids = stratified_split_by_patient(
        patients, config.SPLIT_RATIOS, config.SEED
    )

    split_map = {}
    for p in patients:
        uid = p["uid"]
        if uid in train_uids:
            split_map[uid] = "train"
        elif uid in val_uids:
            split_map[uid] = "val"
        elif uid in test_uids:
            split_map[uid] = "test"

    train_n = len(train_uids)
    val_n = len(val_uids)
    test_n = len(test_uids)
    total_n = train_n + val_n + test_n
    print(
        f"  → train: {train_n} ({100*train_n/total_n:.1f}%) | "
        f"val: {val_n} ({100*val_n/total_n:.1f}%) | "
        f"test: {test_n} ({100*test_n/total_n:.1f}%)"
    )

    # ── 阶段 3: 处理并保存 ──
    print("\n" + "=" * 60)
    print(f"[阶段 3] 处理数据 (float16, 脑+躯干 0–{config.TORSO_START_MM+config.TORSO_LENGTH_MM}mm, 去掉下肢)...")
    print("=" * 60)

    total_saved = 0
    # 按 split 分组处理, 便于显示进度
    for split_name in ("train", "val", "test"):
        split_patients = [
            p for p in patients if split_map[p["uid"]] == split_name
        ]
        if not split_patients:
            continue

        # 按 uid 去重 (同一病人只处理一次, 内部会遍历所有剂量对)
        seen = set()
        unique_split = []
        for p in split_patients:
            if p["uid"] not in seen:
                seen.add(p["uid"])
                unique_split.append(p)

        split_saved = 0
        for patient in tqdm(
            unique_split, desc=f"  [{split_name}]", unit="patient"
        ):
            n = process_patient(patient, config.OUTPUT_DIR, split_name, config)
            split_saved += n

        total_saved += split_saved
        print(f"  [{split_name}] 保存 {split_saved} 个切片")

    print(f"\n[完成] 共保存 {total_saved} 个切片 → {config.OUTPUT_DIR}")
    print(f"        精度: {config.SAVE_DTYPE} | 过滤窗口: 0–{config.TORSO_START_MM+config.TORSO_LENGTH_MM}mm (脑+躯干)")


if __name__ == "__main__":
    main()

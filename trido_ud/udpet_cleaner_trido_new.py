"""
UDPET Cleaner (TriDo v3) — 终极解剖学精准切割版
====================================================
1. 修复脑部缺失：采用绝对高摄取像素计数定位脑实质，向头顶物理回退 120mm 找回头皮。
2. 修复下肢冗余：将截断面积阈值提升至 75%，精准卡在盆底肌，彻底剥离双侧大腿。
3. 严格的 Z 轴降序 (头顶到脚底)。
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

    # ── 解剖学检测核心参数 (v3) ──
    BODY_THRESHOLD = 0.05  # 身体轮廓检测下限
    BRAIN_HOT_SUV = 2.5  # 脑皮层高摄取阈值
    BRAIN_MIN_HOT_PIXELS = 200  # 一层中含有多少个高摄取像素才确认为脑实质 (防手臂干扰)
    HEAD_MARGIN_MM = 120.0  # 从脑实质中心向头顶回退的物理距离 (12cm)

    BODY_AREA_RATIO = 0.75  # 盆底截断阈值 (大腿截面积约占60%，75%可精准卡在骨盆)
    PELVIC_MARGIN_SLICES = 5  # 骨盆下方安全冗余层数 (约1.5cm，保膀胱底)
    BODY_SMOOTH_WINDOW = 5

    SPLIT_RATIOS = (0.70, 0.15, 0.15)
    SEED = 42
    DOSE_MAPPING = {
        "1-2": 2, "d2": 2, "1_2": 2,
        "1-4": 4, "d4": 4, "1_4": 4,
        "1-10": 10, "d10": 10, "1_10": 10,
    }
    SAVE_DTYPE = torch.float16


# ═══════════════════════════════════════════════════════════════════════
#  SUV 计算与解剖边界精准检测
# ═══════════════════════════════════════════════════════════════════════

def calculate_suv_factor(dcm_hdr):
    try:
        weight = float(dcm_hdr.PatientWeight) * 1000.0
        rad_seq = dcm_hdr.RadiopharmaceuticalInformationSequence[0]
        dose = float(rad_seq.RadionuclideTotalDose)
        slope = float(getattr(dcm_hdr, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm_hdr, "RescaleIntercept", 0.0))
        return (slope * weight) / dose, intercept
    except Exception:
        return None, None


def detect_brain_top(slices_sorted, hot_suv=2.5, min_pixels=200, margin_mm=120.0):
    """【重构】无视手臂干扰的脑部定位器"""
    center_idx = 0
    # 1. 从头向下扫描，寻找第一层具有大量高摄取像素的区域（必为脑实质）
    for i, (_z, suv) in enumerate(slices_sorted):
        if np.sum(suv > hot_suv) > min_pixels:
            center_idx = i
            break

    if center_idx == 0:
        return 0

    # 2. 从脑实质向头顶方向（Z轴增大方向，即索引减小）物理回退 120mm
    center_z = slices_sorted[center_idx][0]
    top_idx = center_idx
    for i in range(center_idx - 1, -1, -1):
        if abs(slices_sorted[i][0] - center_z) > margin_mm:
            break
        top_idx = i

    return top_idx


def detect_abdomen_end(slices_sorted, body_threshold=0.05, area_ratio=0.75, smooth_window=5, margin_slices=5):
    """【重构】严格分离大腿的盆底切割器"""
    n = len(slices_sorted)
    if n == 0: return 0

    areas = np.array([np.sum(suv > body_threshold) for _, suv in slices_sorted], dtype=np.float64)

    if smooth_window > 1 and n >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        areas_smooth = np.convolve(areas, kernel, mode='same')
    else:
        areas_smooth = areas

    # 跳过头颈部，寻找躯干的最大横截面积 (通常在肝脏/腹部)
    head_skip = max(1, n // 7)
    max_area = np.max(areas_smooth[head_skip:]) if head_skip < n else np.max(areas_smooth)

    if max_area == 0: return n - 1

    cutoff = max_area * area_ratio

    # 从脚底向上扫描：寻找面积突增突破 75% 的位置（盆底肌分界线）
    for i in range(n - 1, head_skip, -1):
        if areas_smooth[i] > cutoff:
            # 找到盆底后，往下肢方向延伸 margin_slices 层作为安全区
            return min(n - 1, i + margin_slices)

    return n - 1


# ═══════════════════════════════════════════════════════════════════════
#  图像处理 & 病人发现
# ═══════════════════════════════════════════════════════════════════════

def center_crop_numpy(img_array, target_size=256):
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
    return img_array[i: i + th, j: j + tw]


def _is_full_dose_dir(dirpath):
    basename_lower = os.path.basename(dirpath).lower()
    if "full dose" in basename_lower or "100 dose" in basename_lower or "drf_100" in basename_lower:
        return True
    if "normal" in basename_lower:
        return True
    return False


def _is_dicom_dir(dirpath):
    try:
        for f in os.listdir(dirpath):
            if f.lower().endswith((".dcm", ".ima")): return True
    except Exception:
        pass
    return False


def _parse_wb_param(dirname):
    m = re.match(r"[\d.]+\s*x\s*(\d+)\s*wb", dirname.lower().strip())
    if m: return int(m.group(1))
    return None


def _resolve_low_dose_denom(low_dirname, full_dirname, full_param):
    dl = low_dirname.lower().strip()
    for keyword, denom in UDPETCleanerConfig.DOSE_MAPPING.items():
        if keyword in dl and "1000" not in dl:
            return denom, True
    if full_param is not None:
        low_param = _parse_wb_param(low_dirname)
        if low_param is not None and low_param > 0:
            if full_param > low_param and full_param % low_param == 0:
                denom = full_param // low_param
                if denom > 1: return denom, True
    return -1, False


def discover_patients(root_dirs):
    patients = []
    seen_full_dose = set()

    for root_dir in root_dirs:
        if not os.path.exists(root_dir): continue
        for dirpath, dirnames, _ in os.walk(root_dir):
            if not _is_full_dose_dir(dirpath): continue
            full_dose_dir = dirpath
            if full_dose_dir in seen_full_dose: continue
            seen_full_dose.add(full_dose_dir)

            patient_dir = os.path.dirname(full_dose_dir)
            if not os.path.isdir(patient_dir): continue

            full_wb_param = _parse_wb_param(os.path.basename(full_dose_dir))

            try:
                siblings = os.listdir(patient_dir)
            except Exception:
                continue

            low_dose_pairs = []
            for sib in siblings:
                sib_path = os.path.join(patient_dir, sib)
                if not os.path.isdir(sib_path) or sib_path == full_dose_dir: continue
                denom, is_low_dose = _resolve_low_dose_denom(sib, os.path.basename(full_dose_dir), full_wb_param)
                if not is_low_dose: continue
                low_dose_pairs.append((sib_path, denom))

            for sib in siblings:
                sib_path = os.path.join(patient_dir, sib)
                if (not os.path.isdir(sib_path) or sib_path == full_dose_dir
                        or any(sib_path == lp[0] for lp in low_dose_pairs)): continue
                if _is_dicom_dir(sib_path):
                    denom, matched = _resolve_low_dose_denom(sib, os.path.basename(full_dose_dir), full_wb_param)
                    if matched: low_dose_pairs.append((sib_path, denom))

            if not low_dose_pairs: continue

            try:
                dcm_files = [f for f in os.listdir(full_dose_dir) if f.lower().endswith((".dcm", ".ima"))]
                if not dcm_files: continue
                dcm = pydicom.dcmread(os.path.join(full_dose_dir, dcm_files[0]), stop_before_pixels=True)
                uid = getattr(dcm, "StudyInstanceUID", patient_dir)
                gender = getattr(dcm, "PatientSex", "Unknown")
                age_raw = getattr(dcm, "PatientAge", "000Y")
                age = f"{age_raw[:2]}0s" if len(age_raw) >= 2 else "Unknown"
                manufacturer = getattr(dcm, "Manufacturer", "Unknown")

                tracer = "Unknown"
                if "RadiopharmaceuticalInformationSequence" in dcm:
                    tracer = getattr(dcm.RadiopharmaceuticalInformationSequence[0], "Radiopharmaceutical", "Unknown")
                if "FDG" not in tracer.upper() and "FLUORODEOXYGLUCOSE" not in tracer.upper(): continue
            except Exception:
                continue

            patient_id = os.path.basename(patient_dir)
            patients.append({
                "patient_dir": patient_dir, "patient_id": patient_id,
                "full_dose_dir": full_dose_dir, "low_dose_pairs": low_dose_pairs,
                "uid": uid, "gender": gender, "age": age, "manufacturer": manufacturer,
            })
    return patients


# ═══════════════════════════════════════════════════════════════════════
#  分层抽样 & 主处理流程
# ═══════════════════════════════════════════════════════════════════════

def stratified_split_by_patient(patients, ratios, seed):
    random.seed(seed)
    np.random.seed(seed)
    seen_uids = set()
    unique_patients = []
    for p in patients:
        if p["uid"] not in seen_uids:
            seen_uids.add(p["uid"])
            unique_patients.append(p)

    strata = defaultdict(list)
    for p in unique_patients:
        strata[(p["manufacturer"], p["gender"], p["age"])].append(p)

    train_uids, val_uids, test_uids = set(), set(), set()
    for key, group in strata.items():
        n = len(group)
        group.sort(key=lambda x: x["uid"])
        indices = list(range(n))
        np.random.shuffle(indices)

        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1])))
        n_test = max(1, n - n_train - n_val)
        if n >= 3:
            n_val = max(1, n_val);
            n_test = max(1, n_test)
            n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_uids.update(p["uid"] for p in [group[i] for i in indices[:n_train]])
        val_uids.update(p["uid"] for p in [group[i] for i in indices[n_train: n_train + n_val]])
        test_uids.update(p["uid"] for p in [group[i] for i in indices[n_train + n_val:]])

    return train_uids, val_uids, test_uids


def process_patient(patient, output_base, split_name, config):
    full_dir = patient["full_dose_dir"]
    target_size = config.TARGET_SIZE
    save_dtype = config.SAVE_DTYPE

    full_slices = []
    for f in os.listdir(full_dir):
        if not (f.lower().endswith(".dcm") or f.lower().endswith(".ima")): continue
        try:
            dcm = pydicom.dcmread(os.path.join(full_dir, f))
            factor, intercept = calculate_suv_factor(dcm)
            if factor is None: continue
            z_pos = float(dcm.ImagePositionPatient[2])
            suv_array = dcm.pixel_array.astype(np.float32) * factor + intercept
            full_slices.append((z_pos, suv_array))
        except Exception:
            continue

    if not full_slices: return 0

    # 1. 严格 Z 降序 (头顶=大Z -> 脚底=小Z)
    full_slices.sort(key=lambda x: x[0], reverse=True)

    # 2. 检测脑顶 (基于绝对像素计数 + 物理回退)
    brain_top_idx = detect_brain_top(
        full_slices, hot_suv=config.BRAIN_HOT_SUV,
        min_pixels=config.BRAIN_MIN_HOT_PIXELS, margin_mm=config.HEAD_MARGIN_MM
    )

    # 3. 检测大腿切除点 (基于75%面积突变截断)
    body_slices = full_slices[brain_top_idx:]
    abdomen_end_idx = detect_abdomen_end(
        body_slices, body_threshold=config.BODY_THRESHOLD,
        area_ratio=config.BODY_AREA_RATIO, smooth_window=config.BODY_SMOOTH_WINDOW,
        margin_slices=config.PELVIC_MARGIN_SLICES
    )

    torso_full = body_slices[: abdomen_end_idx + 1]
    if not torso_full: return 0

    output_count = 0
    for low_dir, dose_denom in patient["low_dose_pairs"]:
        low_slices = {}
        for f in os.listdir(low_dir):
            if not (f.lower().endswith(".dcm") or f.lower().endswith(".ima")): continue
            try:
                dcm = pydicom.dcmread(os.path.join(low_dir, f))
                factor, intercept = calculate_suv_factor(dcm)
                if factor is None: continue
                z_pos = float(dcm.ImagePositionPatient[2])
                suv_array = dcm.pixel_array.astype(np.float32) * factor + intercept
                low_slices[z_pos] = center_crop_numpy(suv_array, target_size)
            except Exception:
                continue

        if not low_slices: continue
        save_dir = os.path.join(output_base, split_name, patient["patient_id"])
        os.makedirs(save_dir, exist_ok=True)

        for i, (fz, fpx) in enumerate(torso_full):
            matched_z = min((lz for lz in low_slices if abs(lz - fz) < 0.5),
                            key=lambda lz: abs(lz - fz), default=None)
            if matched_z is None: continue

            target_crop = center_crop_numpy(fpx, target_size)
            cond_crop = low_slices[matched_z]

            # 保持 Target 在前 (Tensor 0 = Clean, Tensor 1 = Noise)
            tensor_pair = torch.stack([
                torch.from_numpy(target_crop), torch.from_numpy(cond_crop)
            ]).to(save_dtype)

            save_name = f"{patient['patient_id']}_D{dose_denom}_Z{i:04d}.pt"
            torch.save(tensor_pair, os.path.join(save_dir, save_name), _use_new_zipfile_serialization=True)
            output_count += 1

    return output_count


def main():
    config = UDPETCleanerConfig()
    print("=" * 60);
    print("[阶段 1] 扫描数据源, 构建病人清单...");
    print("=" * 60)
    patients = discover_patients(config.ROOT_DIRS)
    if not patients: return

    unique_uids = len(set(p["uid"] for p in patients))
    print(f"  → 发现 {unique_uids} 位病人, 共 {sum(len(p['low_dose_pairs']) for p in patients)} 组剂量配对")

    print("\n" + "=" * 60);
    print("[阶段 2] 病人级分层抽样 (7:1.5:1.5)...");
    print("=" * 60)
    train_uids, val_uids, test_uids = stratified_split_by_patient(patients, config.SPLIT_RATIOS, config.SEED)
    split_map = {p["uid"]: "train" if p["uid"] in train_uids else "val" if p["uid"] in val_uids else "test" for p in
                 patients}

    print("\n" + "=" * 60);
    print("[阶段 3] 物理截断生成 (找回头皮 + 斩断大腿)...");
    print("=" * 60)
    total_saved = 0
    for split_name in ("train", "val", "test"):
        split_patients = [p for p in patients if split_map[p["uid"]] == split_name]
        if not split_patients: continue
        unique_split = list({p["uid"]: p for p in split_patients}.values())

        split_saved = 0
        for patient in tqdm(unique_split, desc=f"  [{split_name}]", unit="patient"):
            split_saved += process_patient(patient, config.OUTPUT_DIR, split_name, config)
        total_saved += split_saved
        print(f"  [{split_name}] 保存 {split_saved} 个切片")

    print(f"\n[完成] 共保存 {total_saved} 个极致脱水切片 → {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
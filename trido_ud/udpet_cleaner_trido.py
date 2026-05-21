"""
UDPET Cleaner (TriDo Last) — v3 剂量驱动发现 + v3 解剖检测
============================================================
v2 文档骨架 + v3 解剖检测 + DICOM 剂量驱动发现。

核心改进 (vs udpet_cleaner_trido.py):
  1. 剂量发现: 不靠目录名/文件名。遍历所有含 DICOM 的子目录，
     读取 RadionuclideTotalDose 真实剂量值，最大 = Full dose，
     其余 low_dose_pairs 计算 denom = round(full / low)。
  2. 脑部检测: v3 热像素计数(SUV>2.5, >200px) + 120mm 物理回退
  3. 腹部终止: v3 0.75 面积比阈值 + 5层 pelvic margin
  4. 保持 float16、torch.save 无损压缩、7:1.5:1.5 分层、FDG 过滤
  5. 移除 _is_full_dose_dir、_parse_wb_param、_resolve_low_dose_denom、DOSE_MAPPING

┌─ BUGFIX: Tensor 顺序修正 ─────────────────────────────────────┐
│ 原版 cleaner 存 [Clean, Noise] 但 dataset 读 [Noise, Clean]。  │
│ 本版修正为: Tensor[0]=Noise(低剂量条件), Tensor[1]=Clean(目标) │
│ 与 pet_dataset_trido.py 的期望一致。                            │
└────────────────────────────────────────────────────────────────┘

输出结构:
  processed_data_trido/
  ├── train/P0001/P0001_D10_Z0050.pt ...
  ├── val/  P0100/...
  └── test/ P0200/...

兼容: pet_dataset_trido.py 直接读取, Tensor[0]=Noise(条件), Tensor[1]=Clean(目标), float16
"""

import os
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

    # ── v3 解剖学检测核心参数 ──
    # 脑部: 热像素计数 + 物理回退 (防手臂干扰)
    BRAIN_HOT_SUV = 2.5            # 脑皮层高摄取阈值
    BRAIN_MIN_HOT_PIXELS = 200     # 确认脑实质所需的最小高摄取像素数
    HEAD_MARGIN_MM = 120.0         # 从脑实质向头顶回退的物理距离 (12cm)

    # 腹部终止: 0.75截断 + 盆底安全冗余
    BODY_THRESHOLD = 0.05          # 身体轮廓检测下限
    BODY_AREA_RATIO = 0.75         # 盆底截断阈值 (大腿~60%, 75%精准卡骨盆)
    PELVIC_MARGIN_SLICES = 5       # 骨盆下方安全冗余层数 (~1.5cm, 保膀胱底)
    BODY_SMOOTH_WINDOW = 5         # 身体面积平滑窗口 (切片数)

    # ── train/val/test 比例 ──
    SPLIT_RATIOS = (0.70, 0.15, 0.15)
    SEED = 42

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
#  v3 解剖边界精准检测
# ═══════════════════════════════════════════════════════════════════════

def detect_brain_top(slices_sorted, hot_suv=2.5, min_pixels=200, margin_mm=120.0):
    """【v3 热像素方案】无视手臂干扰的脑部定位器。

    原理:
      1. 从头向下扫描，寻找第一层具有大量高摄取像素(SUV > hot_suv)的切片
         → 必为脑实质（手臂注射点的SUV干扰在此阈值下被完全滤除）
      2. 从脑实质中心向头顶方向(Z轴增大方向)物理回退 margin_mm
         → 找回被空气/噪声夹在中间的头皮和颅顶软组织

    对比v2: v2用前景平均SUV>0.4判断，可能在低摄取病人(老年人脑萎缩)
            或高噪声切片上漏检。

    Args:
        slices_sorted: [(z_pos, suv_array), ...] 按 Z 降序排列（头顶在前）
        hot_suv: 高摄取判定阈值 (SUV)
        min_pixels: 确认脑实质所需的最小高摄取像素数
        margin_mm: 向头顶方向回退的物理距离 (mm)

    Returns:
        brain_top_idx: 脑顶切片在 slices_sorted 中的索引
    """
    center_idx = 0

    # Step 1: 从头向下扫描，找脑实质中心
    for i, (_z, suv) in enumerate(slices_sorted):
        if np.sum(suv > hot_suv) > min_pixels:
            center_idx = i
            break

    if center_idx == 0:
        return 0  # 未检测到明显脑组织 → 从第一个切片开始

    # Step 2: 从脑实质向头顶物理回退 margin_mm
    center_z = slices_sorted[center_idx][0]
    top_idx = center_idx
    for i in range(center_idx - 1, -1, -1):
        if abs(slices_sorted[i][0] - center_z) > margin_mm:
            break
        top_idx = i

    return top_idx


def detect_abdomen_end(slices_sorted, body_threshold=0.05, area_ratio=0.75,
                       smooth_window=5, margin_slices=5):
    """【v3 0.75阈值方案】严格分离大腿的盆底切割器。

    原理:
      1. 计算每个切片身体区域的像素数 (SUV > body_threshold)
      2. 滑动平均平滑 → 找躯干最大横截面积(通常在肝脏/腹部)
      3. 从脚底向上扫描: 第一个面积突破 max_area * area_ratio 的位置
         → 盆底肌分界线 (大腿截面积约60%, 75%精准卡在骨盆)
      4. 向下肢方向延伸 margin_slices 层作为膀胱底安全冗余

    对比v2: v2用 area_ratio=0.45 且无margin，大量大腿残留在数据中。

    Args:
        slices_sorted: [(z_pos, suv_array), ...] 按 Z 降序排列
        body_threshold: SUV 超过此值 → 身体组织像素
        area_ratio: 面积低于 max_area * ratio → 判定为大腿
        smooth_window: 滑动平均窗口大小（切片数）
        margin_slices: 骨盆下方安全冗余层数

    Returns:
        abdomen_end_idx: 腹部最后一个切片的索引（含），之后为大腿
    """
    n = len(slices_sorted)
    if n == 0:
        return 0

    # 每个切片的身体面积（像素数）
    areas = np.array([np.sum(suv > body_threshold) for _, suv in slices_sorted],
                     dtype=np.float64)

    # 滑动平均平滑，消除单个切片的噪声波动
    if smooth_window > 1 and n >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        areas_smooth = np.convolve(areas, kernel, mode='same')
    else:
        areas_smooth = areas

    # 跳过头颈部 (前 ~15%)，寻找躯干的最大横截面积
    head_skip = max(1, n // 7)
    if head_skip < n:
        max_area = np.max(areas_smooth[head_skip:])
    else:
        max_area = np.max(areas_smooth)

    if max_area == 0:
        return n - 1

    cutoff = max_area * area_ratio

    # 从脚底向上扫描：寻找面积突增突破 cutoff 的位置（盆底肌分界线）
    for i in range(n - 1, head_skip, -1):
        if areas_smooth[i] > cutoff:
            # 找到盆底后，往下肢方向延伸 margin_slices 层作为安全区
            return min(n - 1, i + margin_slices)

    return n - 1


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
    return img_array[i: i + th, j: j + tw]


# ═══════════════════════════════════════════════════════════════════════
#  病人发现 — 剂量驱动 (无目录名/文件名依赖)
# ═══════════════════════════════════════════════════════════════════════

def _is_dicom_dir(dirpath):
    """Check if a directory contains at least one DICOM file."""
    try:
        for f in os.listdir(dirpath):
            if f.lower().endswith((".dcm", ".ima")):
                return True
    except Exception:
        pass
    return False


def discover_patients(root_dirs):
    """
    Dose-driven patient discovery — 完全不依赖命名约定。

    策略 (自上而下):
      1. Walk 整棵树，找到所有含 DICOM 文件的子目录
      2. 按父目录分组（父目录 = 病人级）
      3. 从每个 DICOM 目录的第一个文件读取 RadionuclideTotalDose
      4. 最大剂量 = Full dose, 其余 = low-dose pairs (denom = round(full/low))
      5. FDG 过滤 + 元数据提取

    对比原版: 不再依赖 'full dose'/'normal'/'drf_100' 等命名模式，
    也不依赖 DOSE_MAPPING 关键词映射或 WB 协议名称解析。

    返回: list of dict {
        patient_dir, patient_id, full_dose_dir,
        low_dose_pairs: [(path, dose_denom), ...],
        uid, gender, age, manufacturer
    }
    """
    patients = []
    seen_patient_dirs = set()

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"  [跳过] 路径不存在: {root_dir}")
            continue

        print(f"  [扫描] {root_dir} ...")

        # ── Step 1: 收集所有含 DICOM 文件的目录，按父目录分组 ──
        # parent_dir → [(subdir_path, first_dcm_path), ...]
        parent_to_dicom_dirs = defaultdict(list)

        for dirpath, _dirnames, filenames in os.walk(root_dir):
            dcm_files = [f for f in filenames
                         if f.lower().endswith((".dcm", ".ima"))]
            if not dcm_files:
                continue

            parent_dir = os.path.dirname(dirpath)
            first_dcm = os.path.join(dirpath, dcm_files[0])
            parent_to_dicom_dirs[parent_dir].append((dirpath, first_dcm))

        if not parent_to_dicom_dirs:
            print(f"    未发现任何 DICOM 目录")
            continue

        # ── Step 2: 逐病人处理 ──
        for patient_dir, dose_entries in parent_to_dicom_dirs.items():
            if patient_dir in seen_patient_dirs:
                continue
            if len(dose_entries) < 2:
                continue  # 需要至少 full + 一个 low-dose

            # Step 3: 读取每个 DICOM 目录的真实剂量
            dose_data = []  # [(dirpath, dose_Bq, dcm_header), ...]
            for subdir_path, first_dcm_path in dose_entries:
                try:
                    dcm = pydicom.dcmread(first_dcm_path, stop_before_pixels=True)
                    rad_seq = dcm.RadiopharmaceuticalInformationSequence[0]
                    dose = float(rad_seq.RadionuclideTotalDose)
                    dose_data.append((subdir_path, dose, dcm))
                except Exception:
                    continue

            if len(dose_data) < 2:
                continue

            # Step 4: 最大剂量 = Full dose
            dose_data.sort(key=lambda x: x[1], reverse=True)
            full_dir, full_dose, full_dcm = dose_data[0]

            # Step 5: 计算 low-dose pairs
            low_dose_pairs = []
            for low_dir, low_dose, _low_dcm in dose_data[1:]:
                if full_dose <= 0 or low_dose <= 0:
                    continue
                denom = round(full_dose / low_dose)
                if denom < 2:
                    continue  # 不是有意义的低剂量
                low_dose_pairs.append((low_dir, denom))

            if not low_dose_pairs:
                continue

            # Step 6: 从 full-dose DICOM 提取元数据 + FDG 过滤
            try:
                uid = getattr(full_dcm, "StudyInstanceUID", patient_dir)
                gender = getattr(full_dcm, "PatientSex", "Unknown")
                age_raw = getattr(full_dcm, "PatientAge", "000Y")
                age = f"{age_raw[:2]}0s" if len(age_raw) >= 2 else "Unknown"
                manufacturer = getattr(full_dcm, "Manufacturer", "Unknown")

                # FDG tracer 检查
                tracer = "Unknown"
                if "RadiopharmaceuticalInformationSequence" in full_dcm:
                    tracer = getattr(
                        full_dcm.RadiopharmaceuticalInformationSequence[0],
                        "Radiopharmaceutical",
                        "Unknown",
                    )
                if "FDG" not in tracer.upper() and "FLUORODEOXYGLUCOSE" not in tracer.upper():
                    continue
            except Exception:
                continue

            seen_patient_dirs.add(patient_dir)
            patient_id = os.path.basename(patient_dir)

            patients.append({
                "patient_dir": patient_dir,
                "patient_id": patient_id,
                "full_dose_dir": full_dir,
                "low_dose_pairs": low_dose_pairs,
                "uid": uid,
                "gender": gender,
                "age": age,
                "manufacturer": manufacturer,
            })

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
            p["uid"] for p in [group[i] for i in indices[n_train: n_train + n_val]]
        )
        test_uids.update(
            p["uid"] for p in [group[i] for i in indices[n_train + n_val:]]
        )

    return train_uids, val_uids, test_uids


# ═══════════════════════════════════════════════════════════════════════
#  主处理逻辑
# ═══════════════════════════════════════════════════════════════════════

def process_patient(patient, output_base, split_name, config):
    """
    处理单个病人的一组 (full-dose, low-dose) 配对。

    v3 流程:
      1. Z 降序排列 (头顶=大Z → 脚底=小Z, DICOM HFS 标准)
      2. 热像素检测脑顶 (SUV>2.5, >200px) + 物理回退120mm找回头皮
      3. 0.75面积比截断腹部 (精准卡盆底肌) + 5层骨盆冗余
      4. Z对齐 full↔low, 输出 float16 .pt 文件

    输出格式: Tensor[0]=Noise(低剂量条件), Tensor[1]=Clean(全剂量目标)
    与 pet_dataset_trido.py 的读取期望一致。
    """
    full_dir = patient["full_dose_dir"]
    target_size = config.TARGET_SIZE
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
            suv_array = dcm.pixel_array.astype(np.float32) * factor + intercept
            full_slices.append((z_pos, suv_array))
        except Exception:
            continue

    if not full_slices:
        return 0

    # ── Step 0: 严格 Z 降序 (头顶=大Z → 脚底=小Z, DICOM HFS 标准) ──
    full_slices.sort(key=lambda x: x[0], reverse=True)

    # ── Step 1: v3 热像素检测脑顶 (无视手臂注射干扰, 物理回退找回头皮) ──
    brain_top_idx = detect_brain_top(
        full_slices,
        hot_suv=config.BRAIN_HOT_SUV,
        min_pixels=config.BRAIN_MIN_HOT_PIXELS,
        margin_mm=config.HEAD_MARGIN_MM,
    )

    # ── Step 2: v3 0.75阈值检测腹部终止 (精准卡盆底肌, 5层膀胱冗余) ──
    body_slices = full_slices[brain_top_idx:]
    abdomen_end_idx = detect_abdomen_end(
        body_slices,
        body_threshold=config.BODY_THRESHOLD,
        area_ratio=config.BODY_AREA_RATIO,
        smooth_window=config.BODY_SMOOTH_WINDOW,
        margin_slices=config.PELVIC_MARGIN_SLICES,
    )

    # ── Step 3: 提取脑顶 → 腹部终止的切片 ──
    torso_full = body_slices[: abdomen_end_idx + 1]

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
                suv_array = dcm.pixel_array.astype(np.float32) * factor + intercept
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

            # [BUGFIX] 顺序: Tensor[0]=Noise(条件), Tensor[1]=Clean(目标)
            # 原版 cleaner 存 [Clean, Noise] 但 dataset 读 [Noise, Clean]，此处修正。
            tensor_pair = torch.stack([
                torch.from_numpy(cond_crop),     # [0] Noise (low-dose condition)
                torch.from_numpy(target_crop),   # [1] Clean (full-dose target)
            ]).to(save_dtype)

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
    print("[阶段 1] 剂量驱动扫描 — 读取 DICOM RadionuclideTotalDose...")
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
    print("[阶段 3] v3解剖检测 — 热像素脑部定位 + 0.75盆底截断...")
    print("  · 剂量: DICOM RadionuclideTotalDose (目录名无关)")
    print("  · 脑部: SUV>2.5 热像素计数 + 120mm 物理回退找回头皮")
    print("  · 腹部: 75% 面积比截断 + 5层膀胱冗余")
    print("  · Z轴: 降序 (头顶=大Z), DICOM HFS 标准")
    print("  · 精度: float16 | 分层: 7:1.5:1.5")
    print("  · 输出: Tensor[0]=Noise, Tensor[1]=Clean (与 dataset 一致)")
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

    print(f"\n[完成] 共保存 {total_saved} 个极致脱水切片 → {config.OUTPUT_DIR}")
    print(f"        精度: {config.SAVE_DTYPE} | 脑部: v3热像素 | 腹部: v3 0.75截断")
    print(f"        剂量: DICOM 驱动 | Tensor: [Noise, Clean]")


if __name__ == "__main__":
    main()

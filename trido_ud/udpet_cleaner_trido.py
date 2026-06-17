"""
UDPET Cleaner (TriDo Last) — v3 目录名驱动剂量发现 + v3 解剖检测
==================================================================
v2 文档骨架 + v3 解剖检测 + 目录名驱动剂量发现。

核心改进 (vs udpet_cleaner_trido.py):
  1. 剂量发现: 不读 RadionuclideTotalDose（注射剂量不变，减少靠 Poisson 仿真）。
     改用目录名推断剂量比例:
       Bern:  Full_dose→full, 1-2→2, 1-4→4, ..., 1-100→100
       Shanghai 2022: 与 Bern 相同 (1-2 dose, 1-4 dose, ...)
       Shanghai 2023: D2/D4/D10/D20/D50/D100 → denom; NORMAL→full
     统一使用 DOSE_MAPPING 关键词匹配 + 长关键字优先排序
  2. 脑部检测: v3 热像素计数(SUV>2.5, >200px) + 120mm 物理回退
  3. 腹部终止: v3 0.75 面积比阈值 + 5层 pelvic margin
  4. 保持 float16、torch.save 无损压缩、7:1.5:1.5 分层、FDG 过滤
  5. body_part 标签: 按切片在 torso 中的相对位置存入 [3,H,W] tensor

┌─ BUGFIX: Tensor 顺序修正 ─────────────────────────────────────┐
│ 原版 cleaner 存 [Clean, Noise] 但 dataset 读 [Noise, Clean]。  │
│ v4 格式: Tensor[0]=body_part(uint8→float16),                   │
│          Tensor[1]=Noise(低剂量条件), Tensor[2]=Clean(目标)     │
│ body_part: 0=brain(<15%), 1=chest(15-50%), 2=abdomen(>50%)     │
│ 与 pet_dataset_trido.py 的 [3,H,W] 期望一致。                   │
└────────────────────────────────────────────────────────────────┘

输出结构:
  processed_data_trido/
  ├── train/P0001/P0001_D10_Z0050.pt ...
  ├── val/  P0100/...
  └── test/ P0200/...

兼容: pet_dataset_trido.py 直接读取, Tensor[0]=body_part, [1]=Noise(条件), [2]=Clean(目标), float16
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
    ]  # 医院数据中心路径（Windows 盘符，部署时需根据实际挂载调整）
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

    DOSE_MAPPING = {
        "1-2": 2, "d2": 2, "1_2": 2,
        "1-4": 4, "d4": 4, "1_4": 4,
        "1-10": 10, "d10": 10, "1_10": 10,
        "1-20": 20, "d20": 20, "1_20": 20,
        "1-50": 50, "d50": 50, "1_50": 50,
        "1-100": 100, "d100": 100, "1_100": 100,
    }


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
    # ⚠️ head_skip 跳过颈部以上，防止脑部高摄取干扰盆底检测
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
#  病人发现 — 自上而下策略
# ═══════════════════════════════════════════════════════════════════════

def _is_full_dose_dir(dirpath):
    """Check if a directory is a full-dose scan directory by its name.

    Strategy A (Bern-Inselspital): basename contains 'full_dose'/'Full_dose' or 'drf_100'
       (NOT '100 dose' — that would falsely match '1-100 dose' low-dose dirs)
    Strategy B (Shanghai-Ruijin):  basename contains 'normal' (WB scan protocol)
       (D2/D4/D10/D100 are LOW-dose, NOT full-dose; only NORMAL marks full-dose)
    """
    basename_lower = os.path.basename(dirpath).lower().replace("_", " ")
    # Bern pattern: "Full_dose" → "full dose", or "drf_100"
    if "full dose" in basename_lower or "drf_100" in basename_lower:
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


def _resolve_low_dose_denom(low_dirname):
    """Determine dose denominator from directory name via DOSE_MAPPING keywords.

    Covers all three naming conventions:
      Bern:       "1-2_dose" → 2, "1-10_dose" → 10, "1-100_dose" → 100
      Shanghai 2022: "1-2 dose" → 2, "1-4 dose" → 4  (matches same keywords)
      Shanghai 2023: "2.886 x 150 WB D4" → 4, "D10" → 10, "D100" → 100

    Returns (denom, is_match) — is_match=False means skip this directory.
    """
    dl = low_dirname.lower().strip()
    for keyword, denom in sorted(UDPETCleanerConfig.DOSE_MAPPING.items(), key=lambda x: -len(x[0])):
        if keyword in dl and "1000" not in dl:
            return denom, True
    return -1, False


def discover_patients(root_dirs):
    """
    扫描所有数据源, 构建病人清单。

    采用「自上而下」策略:
      1. Walk 整棵树, 找到所有 full-dose 目录 (名称含 full_dose/normal/drf_100)
      2. 从父目录(病人级)发现所有低剂量兄弟目录
      3. DOSE_MAPPING 关键词统一解析 (Bern 1-2_dose / Shanghai 2022 1-2 dose / Shanghai 2023 D4)

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

                denom, is_low_dose = _resolve_low_dose_denom(sib)
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
                    denom, matched = _resolve_low_dose_denom(sib)
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
      4. Z对齐 full↔low, 按切片在 torso 中的相对位置计算 body_part:
         i/N < 0.15 → 0(brain), <0.50 → 1(chest), else → 2(abdomen)
      5. 输出 [3,H,W] float16 .pt: [0]=body_part(uint8→float16), [1]=Noise, [2]=Clean
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

    # ── 预创建输出目录（与剂量无关，提到循环外）──
    save_dir = os.path.join(output_base, split_name, patient["patient_id"])
    os.makedirs(save_dir, exist_ok=True)

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

            # ── 🔧 SUV 百分位裁剪: 防止极端热像素 (膀胱 SUV=70+) 导致
            #    后续 patch embedding 尺度失控 → 网格伪影
            #    99.5% percentile: 256×256≈65k像素, 裁剪~327个极端值
            #    (膀胱热区通常100-500像素, 99.9%仅裁65个不够)
            SUV_CLIP_PERCENTILE = 99.5
            pos_mask = target_crop > 0
            vmax_t = np.percentile(target_crop[pos_mask], SUV_CLIP_PERCENTILE) if pos_mask.any() else 1.0
            pos_mask_c = cond_crop > 0
            vmax_c = np.percentile(cond_crop[pos_mask_c], SUV_CLIP_PERCENTILE) if pos_mask_c.any() else 1.0
            vmax = max(vmax_t, vmax_c)

            target_crop = np.clip(target_crop, 0, vmax)
            cond_crop = np.clip(cond_crop, 0, vmax)

            # ── 构建 [3, H, W] tensor: [body_part, Noise, Clean] ──
            # body_part 由切片在 torso 中的相对位置决定
            N = len(torso_full)
            if i < N * 0.15:
                body_part = 0   # brain (头顶 ~15%)
            elif i < N * 0.50:
                body_part = 1   # chest (15%~50%)
            else:
                body_part = 2   # abdomen (50%~脚底)

            bp_tensor = torch.full((target_size, target_size),
                                   body_part, dtype=torch.float32)

            tensor_pair = torch.stack([
                bp_tensor,                       # [0] body_part label
                torch.from_numpy(cond_crop),     # [1] Noise (low-dose condition)
                torch.from_numpy(target_crop),   # [2] Clean (full-dose target)
            ]).to(save_dtype)  # float32 → save_dtype (float16)，非 uint8 自动提升

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
    print("[阶段 1] 目录名驱动扫描 — 从目录名推断剂量比例...")
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
    print("  · 剂量: 目录名推断 (Bern: 1-X, Shanghai: D关键词)")
    print("  · 脑部: SUV>2.5 热像素计数 + 120mm 物理回退找回头皮")
    print("  · 腹部: 75% 面积比截断 + 5层膀胱冗余")
    print("  · Z轴: 降序 (头顶=大Z), DICOM HFS 标准")
    print("  · 精度: float16 | 分层: 7:1.5:1.5")
    print("  · 输出: Tensor[0]=body_part, [1]=Noise, [2]=Clean (与 dataset 一致)")
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
    print(f"        剂量: 目录名驱动 | Tensor: [body_part, Noise, Clean]")


if __name__ == "__main__":
    main()

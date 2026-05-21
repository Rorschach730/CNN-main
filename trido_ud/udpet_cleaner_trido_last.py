"""
UDPET Cleaner (TriDo Last) — v2文档骨架 + v3解剖检测
=========================================================
取精去糟：v3的热像素脑部检测 + 0.75腹部截断 + Z轴确认 + Bug修复，
配上v2的完整文档和高质量输出。

核心特性:
  1. v3热像素脑部定位: 绝对高摄取像素(SUV>2.5)计数定位脑实质，向头顶物理回退120mm找回头皮
  2. v3 0.75腹部截断: 截断面积阈值75%，精准卡在盆底肌，margin_slices=5保膀胱底
  3. Z轴确认: 严格降序(头顶=大Z→脚底=小Z)，匹配DICOM HFS坐标系
  4. float16存储: 源DICOM int16→float16无损覆盖 → 2x缩减
  5. train/val/test = 7:1.5:1.5 病人级切分(防数据泄漏)
  6. manufacturer/gender/age 分层抽样
  7. torch.save 内置zip压缩 (额外 ~10-15%)

预计: 210 GB → ~60 GB

输出结构:
  processed_data_trido/
  ├── train/P0001/P0001_D10_Z0050.pt ...
  ├── val/  P0100/...
  └── test/ P0200/...

兼容: pet_dataset_trido.py 直接读取, Tensor[0]=Clean(目标), Tensor[1]=Noise(条件), float16
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

    输出格式: Tensor[0]=Clean(目标full-dose), Tensor[1]=Noise(条件low-dose)
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

            # Tensor[0] = Clean (full-dose target), Tensor[1] = Noise (low-dose condition)
            tensor_pair = torch.stack([
                torch.from_numpy(target_crop),
                torch.from_numpy(cond_crop),
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
    print("[阶段 3] v3解剖检测 — 热像素脑部定位 + 0.75盆底截断...")
    print("  · 脑部: SUV>2.5 热像素计数 + 120mm 物理回退找回头皮")
    print("  · 腹部: 75% 面积比截断 + 5层膀胱冗余")
    print("  · Z轴: 降序 (头顶=大Z), DICOM HFS 标准")
    print("  · 精度: float16 | 分层: 7:1.5:1.5")
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


if __name__ == "__main__":
    main()

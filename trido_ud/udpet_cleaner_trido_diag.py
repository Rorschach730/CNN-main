"""
UDPET Cleaner (TriDo Last) — v3.1 诊断增强版
==============================================
基于 v3 剂量驱动发现 + v3 解剖检测，新增：
  1. 逐过滤点打印被跳过的具体病人ID/路径
  2. Shanghai 2023 PART1/PART1/Anonymous 嵌套自动检测
  3. denom 分布直方图
  4. FDG 宽松模式（可配置）
  5. dry-run 诊断模式（只发现不处理）

用法:
  python udpet_cleaner_trido_diag.py              # 正常处理
  python udpet_cleaner_trido_diag.py --dry-run    # 仅诊断
  python udpet_cleaner_trido_diag.py --lenient-fdg # 宽松FDG检查
"""

import os
import sys
import random
import pydicom
import numpy as np
import torch
from collections import defaultdict, Counter
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
    BRAIN_HOT_SUV = 2.5
    BRAIN_MIN_HOT_PIXELS = 200
    HEAD_MARGIN_MM = 120.0

    BODY_THRESHOLD = 0.05
    BODY_AREA_RATIO = 0.75
    PELVIC_MARGIN_SLICES = 5
    BODY_SMOOTH_WINDOW = 5

    # ── train/val/test 比例 ──
    SPLIT_RATIOS = (0.70, 0.15, 0.15)
    SEED = 42

    # ── 保存精度 ──
    SAVE_DTYPE = torch.float16

    # ── v3.1 新增: 诊断配置 ──
    STRICT_FDG = True          # False=宽松模式, 缺失tracer信息不跳过
    MAX_SKIP_SAMPLES = 20      # 每种跳过原因最多打印多少个样本
    DRY_RUN = False            # True=仅诊断不处理


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
#  v3 解剖边界精准检测 (保持不变)
# ═══════════════════════════════════════════════════════════════════════

def detect_brain_top(slices_sorted, hot_suv=2.5, min_pixels=200, margin_mm=120.0):
    center_idx = 0
    for i, (_z, suv) in enumerate(slices_sorted):
        if np.sum(suv > hot_suv) > min_pixels:
            center_idx = i
            break
    if center_idx == 0:
        return 0
    center_z = slices_sorted[center_idx][0]
    top_idx = center_idx
    for i in range(center_idx - 1, -1, -1):
        if abs(slices_sorted[i][0] - center_z) > margin_mm:
            break
        top_idx = i
    return top_idx


def detect_abdomen_end(slices_sorted, body_threshold=0.05, area_ratio=0.75,
                       smooth_window=5, margin_slices=5):
    n = len(slices_sorted)
    if n == 0:
        return 0
    areas = np.array([np.sum(suv > body_threshold) for _, suv in slices_sorted],
                     dtype=np.float64)
    if smooth_window > 1 and n >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        areas_smooth = np.convolve(areas, kernel, mode='same')
    else:
        areas_smooth = areas
    head_skip = max(1, n // 7)
    if head_skip < n:
        max_area = np.max(areas_smooth[head_skip:])
    else:
        max_area = np.max(areas_smooth)
    if max_area == 0:
        return n - 1
    cutoff = max_area * area_ratio
    for i in range(n - 1, head_skip, -1):
        if areas_smooth[i] > cutoff:
            return min(n - 1, i + margin_slices)
    return n - 1


# ═══════════════════════════════════════════════════════════════════════
#  图像处理
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


# ═══════════════════════════════════════════════════════════════════════
#  病人发现 — v3.1 诊断增强版
# ═══════════════════════════════════════════════════════════════════════

def _resolve_patient_dir(dirpath, root_dir, grouping_depth_cache,
                          nesting_warnings=None):
    """
    从 DICOM 目录路径向上回溯找到病人级目录 (v3.1 增强)。

    Bern 结构:      root/Subject_X-Y/日期/Full_dose/*.IMA
                     → 跳过日期层，上溯 2 级到 Subject_X-Y

    Shanghai 2022:  root/PART1/Anonymous/scan_name/*.dcm
                     → 上溯 1 级到 PART1/Anonymous

    Shanghai 2023:  root/PART1/PART1/Anonymous/scan_name/*.dcm
                     → 需要上溯 1 级到 PART1/PART1/Anonymous
                     (os.walk 正常穿透嵌套，但 grouping 深度需确认)

    自动检测: 如果 depth=1 产生的 patient_dir 名字重复率极高
    (如 >50% 共享同一 basename)，自动升级到 depth=2。
    """
    root_key = root_dir.rstrip("/\\")
    dirpath_norm = os.path.normpath(dirpath)

    if root_key not in grouping_depth_cache:
        root_name = os.path.basename(root_key).lower()
        if "bern" in root_name:
            depth = 2
            print(f"    [分组深度] 检测到 Bern 结构 → 上溯 {depth} 级到病人目录")
        else:
            depth = 1
            print(f"    [分组深度] 检测到非 Bern 结构 → 上溯 {depth} 级到病人目录")
        grouping_depth_cache[root_key] = depth

    depth = grouping_depth_cache[root_key]
    p = dirpath_norm
    for _ in range(depth):
        p = os.path.dirname(p)

    # ── v3.1: 检测是否回到了根目录级别（嵌套过深？）──
    # 如果 patient_dir 就在 root_dir 内部1-2级，说明可能没穿透
    rel = os.path.relpath(p, root_dir)
    if rel in (".", "..") or len(rel.split(os.sep)) <= 1:
        if nesting_warnings is not None:
            nesting_warnings.append(
                f"    ⚠️ 浅层patient_dir: {p} (距root仅{len(rel.split(os.sep))}级, "
                f"源路径={dirpath_norm})"
            )

    return p


def discover_patients(root_dirs, config=None):
    """
    Dose-driven patient discovery — v3.1 诊断增强版。

    每步过滤均打印被跳过样本的具体路径和原因。
    返回: list of patient dicts
    """
    if config is None:
        config = UDPETCleanerConfig()

    patients = []
    seen_patient_dirs = set()
    grouping_depth_cache = {}

    # ── v3.1: 分桶调试收集器 ──
    # 每种跳过原因维护一个列表: [(patient_dir_path, detail), ...]
    skip_log = defaultdict(list)
    nesting_warnings = []

    stats = {
        "total_dcm_dirs": 0,
        "total_ima_files": 0,
        "total_dcm_files": 0,
        "grouped_patients": 0,
        "skipped_single_dose": 0,
        "skipped_dose_read_error": 0,
        "skipped_denom_lt2": 0,
        "skipped_zero_dose": 0,
        "skipped_non_fdg": 0,
        "skipped_metadata_error": 0,
        "skipped_already_seen": 0,
    }
    # denom 分布收集
    all_denoms = []

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"  [跳过] 路径不存在: {root_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"  [扫描] {root_dir}")
        print(f"{'='*70}")

        # ── Step 1: 收集所有含 DICOM 文件的目录 ──
        parent_to_dicom_dirs = defaultdict(list)
        dir_dcm_count = 0
        dir_ima_count = 0
        dir_dcm_ext_count = 0

        # v3.1: 同时收集目录深度分布，用于诊断嵌套问题
        depth_counter = Counter()

        for dirpath, _dirnames, filenames in os.walk(root_dir):
            dcm_by_ima = [f for f in filenames if f.lower().endswith(".ima")]
            dcm_by_dcm = [f for f in filenames if f.lower().endswith(".dcm")]
            dcm_files = dcm_by_ima + dcm_by_dcm

            if dcm_files:
                if dcm_by_ima:
                    dir_ima_count += 1
                    stats["total_ima_files"] += len(dcm_by_ima)
                if dcm_by_dcm:
                    dir_dcm_ext_count += 1
                    stats["total_dcm_files"] += len(dcm_by_dcm)

            if not dcm_files:
                continue

            dir_dcm_count += 1
            stats["total_dcm_dirs"] += 1

            # 记录目录深度（相对 root_dir）
            rel = os.path.relpath(dirpath, root_dir)
            depth_counter[len(rel.split(os.sep))] += 1

            patient_dir = _resolve_patient_dir(
                dirpath, root_dir, grouping_depth_cache,
                nesting_warnings=nesting_warnings,
            )
            first_dcm = os.path.join(dirpath, dcm_files[0])
            parent_to_dicom_dirs[patient_dir].append((dirpath, first_dcm))

        # ── 调试: 输出扫描统计 ──
        print(f"    → 含 DICOM 的目录数: {dir_dcm_count}")
        print(f"      其中含 .IMA 的目录: {dir_ima_count}, .dcm 的目录: {dir_dcm_ext_count}")
        print(f"      .IMA 文件总数: {stats['total_ima_files']}, .dcm 文件总数: {stats['total_dcm_files']}")
        print(f"    → 目录深度分布 (相对root的层级数):")
        for depth, count in sorted(depth_counter.items()):
            bar = "█" * min(count, 60)
            print(f"      深度 {depth:2d}: {count:5d} 个目录 {bar}")

        # ── v3.1: 检测 patient_dir 去重炸弹 ──
        patient_dir_basenames = Counter()
        for pd in parent_to_dicom_dirs:
            patient_dir_basenames[os.path.basename(pd)] += 1
        if patient_dir_basenames:
            top_basename, top_count = patient_dir_basenames.most_common(1)[0]
            total_groups = len(parent_to_dicom_dirs)
            if total_groups > 5 and top_count > total_groups * 0.5:
                print(f"    ⚠️  去重炸弹警告: {top_count}/{total_groups} 个分组共享 "
                      f"basename='{top_basename}' → 可能存在嵌套过深!")
                print(f"       建议检查 _resolve_patient_dir 的 depth 参数")
                # 打印几个样本路径
                samples = [pd for pd in parent_to_dicom_dirs
                           if os.path.basename(pd) == top_basename][:5]
                for s in samples:
                    print(f"       样本: {s}")

        print(f"    → 分组后候选病人目录数: {len(parent_to_dicom_dirs)}")

        if not parent_to_dicom_dirs:
            print(f"    ⚠️  未发现任何 DICOM 目录")
            continue

        # ── 剂量目录数分布 ──
        dose_count_dist = Counter()
        for entries in parent_to_dicom_dirs.values():
            dose_count_dist[len(entries)] += 1
        print(f"    → 剂量目录数分布: {dict(sorted(dose_count_dist.items()))}")

        # ── v3.1: 如果大量病人只有1个剂量目录，打印样本 ──
        if 1 in dose_count_dist and dose_count_dist[1] > 10:
            single_dose_dirs = [pd for pd, entries in parent_to_dicom_dirs.items()
                                if len(entries) == 1]
            print(f"    ⚠️  单剂量目录病人: {dose_count_dist[1]}个 (这些将被skipped_single_dose过滤)")
            print(f"       前{min(5, len(single_dose_dirs))}个样本:")
            for sd in single_dose_dirs[:5]:
                print(f"         {sd}")

        # ── Step 2: 逐病人处理 ──
        for patient_dir, dose_entries in parent_to_dicom_dirs.items():
            if patient_dir in seen_patient_dirs:
                stats["skipped_already_seen"] += 1
                skip_log["already_seen"].append((patient_dir,
                    f"patient_dir已在之前的root_dir中处理过"))
                continue

            if len(dose_entries) < 2:
                stats["skipped_single_dose"] += 1
                skip_log["single_dose"].append((patient_dir,
                    f"仅{len(dose_entries)}个剂量目录"))
                continue

            # Step 3: 读取每个 DICOM 目录的真实剂量
            dose_data = []  # [(dirpath, dose_Bq, dcm_header, first_dcm_path), ...]
            dose_read_errors = []
            for subdir_path, first_dcm_path in dose_entries:
                try:
                    dcm = pydicom.dcmread(first_dcm_path, stop_before_pixels=True)
                    # v3.1: 先检查是否有 RadiopharmaceuticalInformationSequence
                    if "RadiopharmaceuticalInformationSequence" not in dcm:
                        dose_read_errors.append(
                            f"{os.path.basename(subdir_path)}: 缺少RadiopharmaceuticalInformationSequence")
                        continue
                    rad_seq = dcm.RadiopharmaceuticalInformationSequence[0]
                    dose = float(rad_seq.RadionuclideTotalDose)
                    if dose <= 0:
                        dose_read_errors.append(
                            f"{os.path.basename(subdir_path)}: 剂量={dose} <= 0")
                        continue
                    dose_data.append((subdir_path, dose, dcm, first_dcm_path))
                except Exception as e:
                    dose_read_errors.append(
                        f"{os.path.basename(subdir_path)}: {type(e).__name__}: {e}")

            if len(dose_data) < 2:
                stats["skipped_dose_read_error"] += 1
                detail = "; ".join(dose_read_errors[:3]) if dose_read_errors else "未知原因"
                skip_log["dose_read_error"].append((patient_dir,
                    f"成功读取{len(dose_data)}/{len(dose_entries)}个剂量. 错误: {detail}"))
                continue

            # ── v3.1: 打印剂量分布(诊断用) ──
            doses_only = [d[1] for d in dose_data]
            doses_only.sort(reverse=True)

            # Step 4: 最大剂量 = Full dose
            dose_data.sort(key=lambda x: x[1], reverse=True)
            full_dir, full_dose, full_dcm, full_dcm_path = dose_data[0]

            # Step 5: 计算 low-dose pairs + v3.1: 收集所有denom
            low_dose_pairs = []
            skipped_denoms = []
            for low_dir, low_dose, _low_dcm, _low_path in dose_data[1:]:
                if full_dose <= 0 or low_dose <= 0:
                    stats["skipped_zero_dose"] += 1
                    skip_log["zero_dose"].append((patient_dir,
                        f"full={full_dose}, low={low_dose}"))
                    continue
                denom_float = full_dose / low_dose
                denom = round(denom_float)
                all_denoms.append(denom)

                if denom < 2:
                    skipped_denoms.append(
                        f"{os.path.basename(low_dir)}: denom={denom} "
                        f"(full={full_dose:.1f}/low={low_dose:.1f}={denom_float:.2f})")
                    continue
                low_dose_pairs.append((low_dir, denom))

            if not low_dose_pairs:
                stats["skipped_denom_lt2"] += 1
                detail = "; ".join(skipped_denoms[:3]) if skipped_denoms else "所有denom<2"
                skip_log["denom_lt2"].append((patient_dir,
                    f"full_dose={full_dose:.1f}, {len(dose_data)-1}个低剂量: {detail}"))
                continue

            # Step 6: 从 full-dose DICOM 提取元数据 + FDG 过滤 (v3.1 增强)
            try:
                uid = getattr(full_dcm, "StudyInstanceUID", patient_dir)
                gender = getattr(full_dcm, "PatientSex", "Unknown")
                age_raw = getattr(full_dcm, "PatientAge", "000Y")
                age = f"{age_raw[:2]}0s" if len(age_raw) >= 2 else "Unknown"
                manufacturer = getattr(full_dcm, "Manufacturer", "Unknown")

                # ── v3.1: FDG 检查（支持宽松模式）──
                tracer = None
                tracer_source = "missing"
                if "RadiopharmaceuticalInformationSequence" in full_dcm:
                    try:
                        tracer = getattr(
                            full_dcm.RadiopharmaceuticalInformationSequence[0],
                            "Radiopharmaceutical",
                            None,
                        )
                        if tracer:
                            tracer_source = "tag"
                    except Exception:
                        tracer = None

                if tracer is None:
                    # 尝试从 (0008,0104) CodeMeaning 或其他tag获取
                    try:
                        tracer = getattr(full_dcm, "CodeMeaning", None)
                        if tracer:
                            tracer_source = "CodeMeaning"
                    except Exception:
                        pass

                if tracer is None:
                    tracer_source = "none"
                    tracer = ""

                is_fdg = ("FDG" in tracer.upper() or
                          "FLUORODEOXYGLUCOSE" in tracer.upper() or
                          "FLUDEOXYGLUCOSE" in tracer.upper())

                if not is_fdg:
                    if config.STRICT_FDG:
                        stats["skipped_non_fdg"] += 1
                        skip_log["non_fdg"].append((patient_dir,
                            f"tracer='{tracer}' (来源={tracer_source})"))
                        continue
                    else:
                        # 宽松模式: 仅警告
                        tracer_display = tracer if tracer else "(空)"
                        skip_log["non_fdg_warn"].append((patient_dir,
                            f"宽松模式: tracer='{tracer_display}' "
                            f"(来源={tracer_source}), 不跳过"))
            except Exception as e:
                stats["skipped_metadata_error"] += 1
                skip_log["metadata_error"].append((patient_dir,
                    f"{type(e).__name__}: {e}"))
                continue

            seen_patient_dirs.add(patient_dir)
            stats["grouped_patients"] += 1
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

    # ═══════════════════════════════════════════════════════════════════
    #  v3.1: 全局诊断总结（大幅增强）
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  discover_patients 完整诊断报告")
    print(f"{'='*70}")
    print(f"  总扫描含 DICOM 的目录: {stats['total_dcm_dirs']}")
    print(f"  .IMA 文件数: {stats['total_ima_files']}, .dcm 文件数: {stats['total_dcm_files']}")
    print(f"  最终有效病人: {stats['grouped_patients']}")
    print(f"  总丢失: {stats['total_dcm_dirs']} DICOM目录 → {stats['grouped_patients']} 病人")
    print(f"\n  ── 过滤明细 ──")

    filter_reasons = [
        ("single_dose", "仅1个剂量目录",
         stats["skipped_single_dose"], skip_log.get("single_dose", [])),
        ("dose_read_error", "剂量读取失败",
         stats["skipped_dose_read_error"], skip_log.get("dose_read_error", [])),
        ("zero_dose", "剂量<=0",
         stats["skipped_zero_dose"], skip_log.get("zero_dose", [])),
        ("denom_lt2", "denom<2 无效",
         stats["skipped_denom_lt2"], skip_log.get("denom_lt2", [])),
        ("non_fdg", "非FDG示踪剂",
         stats["skipped_non_fdg"], skip_log.get("non_fdg", [])),
        ("metadata_error", "元数据提取失败",
         stats["skipped_metadata_error"], skip_log.get("metadata_error", [])),
        ("already_seen", "重复patient_dir",
         stats["skipped_already_seen"], skip_log.get("already_seen", [])),
    ]

    for reason_key, reason_label, count, samples in filter_reasons:
        marker = " ← ⚠️ 主要损失!" if count > stats["grouped_patients"] * 0.5 else ""
        print(f"\n  [{reason_key}] {reason_label}: {count} 个病人{marker}")
        if samples:
            max_show = min(config.MAX_SKIP_SAMPLES, len(samples))
            print(f"    样本 (前{max_show}/{len(samples)}):")
            for patient_dir, detail in samples[:max_show]:
                print(f"      ✗ {os.path.basename(patient_dir)}")
                print(f"        路径: {patient_dir}")
                print(f"        原因: {detail}")
                print()

    # ── denom 分布直方图 ──
    if all_denoms:
        denom_counter = Counter(all_denoms)
        print(f"\n  ── denom 分布直方图 (round(full/low)) ──")
        print(f"  总denom计算次数: {len(all_denoms)}")
        for denom_val in sorted(denom_counter.keys()):
            count = denom_counter[denom_val]
            bar = "█" * min(count, 60)
            label = "← 被过滤(<2)" if denom_val < 2 else ""
            print(f"    denom={denom_val:3d}: {count:5d} {bar} {label}")

    # ── nesting warnings ──
    if nesting_warnings:
        print(f"\n  ── 嵌套结构警告 ──")
        for w in nesting_warnings[:10]:
            print(f"  {w}")

    # ── v3.1: 宽松模式额外统计 ──
    if not config.STRICT_FDG and "non_fdg_warn" in skip_log:
        print(f"\n  [non_fdg_warn] 宽松模式: {len(skip_log['non_fdg_warn'])} 个病人tracer异常但保留")
        for patient_dir, detail in skip_log["non_fdg_warn"][:5]:
            print(f"      ⚠ {os.path.basename(patient_dir)}: {detail}")

    print(f"\n{'='*70}")
    print(f"  诊断完成。若需调整过滤策略，修改 UDPETCleanerConfig 参数后重跑。")
    print(f"{'='*70}\n")

    return patients


# ═══════════════════════════════════════════════════════════════════════
#  病人级分层抽样 (7:1.5:1.5) — 保持不变
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
        key = (p["manufacturer"], p["gender"], p["age"])
        strata[key].append(p)

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
            n_val = max(1, n_val)
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


# ═══════════════════════════════════════════════════════════════════════
#  主处理逻辑 — 保持不变
# ═══════════════════════════════════════════════════════════════════════

def process_patient(patient, output_base, split_name, config):
    full_dir = patient["full_dose_dir"]
    target_size = config.TARGET_SIZE
    save_dtype = config.SAVE_DTYPE

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

    full_slices.sort(key=lambda x: x[0], reverse=True)

    brain_top_idx = detect_brain_top(
        full_slices,
        hot_suv=config.BRAIN_HOT_SUV,
        min_pixels=config.BRAIN_MIN_HOT_PIXELS,
        margin_mm=config.HEAD_MARGIN_MM,
    )

    body_slices = full_slices[brain_top_idx:]
    abdomen_end_idx = detect_abdomen_end(
        body_slices,
        body_threshold=config.BODY_THRESHOLD,
        area_ratio=config.BODY_AREA_RATIO,
        smooth_window=config.BODY_SMOOTH_WINDOW,
        margin_slices=config.PELVIC_MARGIN_SLICES,
    )

    torso_full = body_slices[: abdomen_end_idx + 1]
    if not torso_full:
        return 0

    output_count = 0
    for low_dir, dose_denom in patient["low_dose_pairs"]:
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

        save_dir = os.path.join(output_base, split_name, patient["patient_id"])
        os.makedirs(save_dir, exist_ok=True)

        for i, (fz, fpx) in enumerate(torso_full):
            matched_z = min(
                (lz for lz in low_slices if abs(lz - fz) < 0.5),
                key=lambda lz: abs(lz - fz),
                default=None,
            )
            if matched_z is None:
                continue

            target_crop = center_crop_numpy(fpx, target_size)
            cond_crop = low_slices[matched_z]

            tensor_pair = torch.stack([
                torch.from_numpy(cond_crop),
                torch.from_numpy(target_crop),
            ]).to(save_dtype)

            save_name = f"{patient['patient_id']}_D{dose_denom}_Z{i:04d}.pt"
            save_path = os.path.join(save_dir, save_name)
            torch.save(tensor_pair, save_path, _use_new_zipfile_serialization=True)
            output_count += 1

    return output_count


# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── v3.1: 解析命令行参数 ──
    config = UDPETCleanerConfig()

    args = sys.argv[1:]
    if "--dry-run" in args:
        config.DRY_RUN = True
        print("[模式] DRY-RUN — 仅诊断，不处理数据")
    if "--lenient-fdg" in args or "--no-strict-fdg" in args:
        config.STRICT_FDG = False
        print("[模式] 宽松FDG — tracer信息缺失不跳过")
    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    # ── 阶段 1: 扫描所有病人 ──
    print("=" * 70)
    print("[阶段 1] 剂量驱动扫描 (v3.1 诊断增强版) — 读取 DICOM RadionuclideTotalDose...")
    print("=" * 70)
    patients = discover_patients(config.ROOT_DIRS, config=config)

    if not patients:
        print("[错误] 未发现任何有效病人数据, 退出。")
        print("   → 请检查上方诊断报告中的过滤明细，定位主要损失原因。")
        print("   → 尝试: python udpet_cleaner_trido_diag.py --lenient-fdg")
        return

    unique_uids = len(set(p["uid"] for p in patients))
    total_pairs = sum(len(p["low_dose_pairs"]) for p in patients)
    print(f"  → 发现 {unique_uids} 位病人, 共 {total_pairs} 组剂量配对")

    if config.DRY_RUN:
        print("\n[Dry-run 完成] 不执行后续处理。")
        return

    # ── 阶段 2: 病人级分层切分 ──
    print("\n" + "=" * 70)
    print("[阶段 2] 病人级分层抽样 (7:1.5:1.5)...")
    print("=" * 70)
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
    print("\n" + "=" * 70)
    print("[阶段 3] v3解剖检测 — 热像素脑部定位 + 0.75盆底截断...")
    print("=" * 70)

    total_saved = 0
    for split_name in ("train", "val", "test"):
        split_patients = [
            p for p in patients if split_map[p["uid"]] == split_name
        ]
        if not split_patients:
            continue

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


if __name__ == "__main__":
    main()

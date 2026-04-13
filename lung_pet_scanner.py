import os
import pydicom
from collections import Counter
import sys
import time


def fast_lung_dicom_scanner(root_dir):
    print("=" * 70)
    print(f"[*] 启动 Lung-PET-CT-Dx 极速双模态物理探针 (含空间分辨率增强)")
    print(f"[*] 扫描目录: {root_dir}")
    print(f"[*] 警告: 探针已切断像素流，仅执行底层元数据剥离。\n" + "=" * 70)

    if not os.path.exists(root_dir):
        print(f"[!!!] 致命错误: 数据总干路径不存在 -> {root_dir}")
        return

    stats = {
        'PT': {
            'patients': set(),
            'gender': Counter(),
            'age': Counter(),
            'image_size': Counter(),
            'tracer': Counter(),
            'slice_thickness': Counter(),  # [新增] Z轴层厚
            'pixel_spacing': Counter()  # [新增] XY平面物理分辨率
        },
        'CT': {
            'patients': set(),
            'gender': Counter(),
            'age': Counter(),
            'image_size': Counter(),
            'slice_thickness': Counter(),
            'pixel_spacing': Counter()
        }
    }

    processed_series = set()
    total_files_scanned = 0

    for dirpath, _, filenames in os.walk(root_dir):
        dicom_files = [f for f in filenames if f.lower().endswith(('.dcm', '.ima'))]
        if not dicom_files:
            continue

        sample_file = os.path.join(dirpath, dicom_files[0])

        try:
            dcm = pydicom.dcmread(sample_file, stop_before_pixels=True)

            series_uid = getattr(dcm, 'SeriesInstanceUID', dirpath)
            if series_uid in processed_series:
                continue
            processed_series.add(series_uid)

            modality = getattr(dcm, 'Modality', 'Unknown')
            if modality not in ['PT', 'CT']:
                continue

            patient_id = getattr(dcm, 'PatientID', 'Unknown')

            time.sleep(0.005)
            total_files_scanned += len(dicom_files)

            sys.stdout.write(
                f"\r    -> 物理寻址中... 已锁定 PET: {len(stats['PT']['patients']):>4} 例 | CT: {len(stats['CT']['patients']):>4} 例 ")
            sys.stdout.flush()

            gender = getattr(dcm, 'PatientSex', 'Unknown')
            age_raw = getattr(dcm, 'PatientAge', 'Unknown')
            age_group = f"{age_raw[:2]}0s" if age_raw != 'Unknown' and len(age_raw) >= 2 else 'Unknown'
            rows = getattr(dcm, 'Rows', 'Unknown')
            cols = getattr(dcm, 'Columns', 'Unknown')
            size = f"{rows}x{cols}"

            target_stat = stats[modality]
            if patient_id not in target_stat['patients']:
                target_stat['patients'].add(patient_id)
                target_stat['gender'][gender] += 1
                target_stat['age'][age_group] += 1

            target_stat['image_size'][size] += 1

            # --- [新增] 空间分辨率联合提取 ---
            # 1. 提取 Z 轴层厚 (Slice Thickness)
            thickness = getattr(dcm, 'SliceThickness', 'Unknown')
            if thickness != 'Unknown':
                # 保留两位小数，统一物理精度
                thickness_str = f"{float(thickness):.2f} mm"
            else:
                thickness_str = "Unknown"
            target_stat['slice_thickness'][thickness_str] += 1

            # 2. 提取 XY 平面像素间距 (Pixel Spacing)
            spacing = getattr(dcm, 'PixelSpacing', 'Unknown')
            if spacing != 'Unknown' and isinstance(spacing, pydicom.multival.MultiValue):
                # PixelSpacing 通常是 [y, x] 格式
                spacing_str = f"{float(spacing[0]):.3f} x {float(spacing[1]):.3f} mm"
            else:
                spacing_str = "Unknown"
            target_stat['pixel_spacing'][spacing_str] += 1
            # -------------------------------

            if modality == 'PT':
                tracer = 'Unknown'
                if 'RadiopharmaceuticalInformationSequence' in dcm:
                    seq = dcm.RadiopharmaceuticalInformationSequence[0]
                    tracer = getattr(seq, 'Radiopharmaceutical', 'Unknown')
                target_stat['tracer'][tracer] += 1

        except Exception:
            continue

    print("\n\n" + "=" * 70)
    print(f"[*] 全局扫描终了。累计穿透底层切片数: {total_files_scanned}")
    print("=" * 70)

    for mod in ['PT', 'CT']:
        mod_name = "正电子发射断层扫描 (PET)" if mod == 'PT' else "计算机断层扫描 (CT)"
        print(f"\n>>> 模态: {mod_name} | 独立患者总数: {len(stats[mod]['patients'])} <<<")
        print("-" * 50)

        categories = ['gender', 'age', 'image_size', 'slice_thickness', 'pixel_spacing']
        if mod == 'PT': categories.insert(3, 'tracer')

        for cat in categories:
            print(f"  [+] {cat.upper()} 分布:")
            for key, val in stats[mod][cat].most_common():
                print(f"      - {str(key):<20}: {val:>5} 序列")
    print("=" * 70)


if __name__ == "__main__":
    LUNG_PET_DIR = "./lung_pet_data"
    fast_lung_dicom_scanner(LUNG_PET_DIR)
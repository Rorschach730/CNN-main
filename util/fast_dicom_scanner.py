import os
import pydicom
from collections import Counter
import sys
import time


def fast_dicom_scanner(root_dirs):
    print(f"[*] 启动极速 DICOM 头文件探针 (UID绝对寻址 + I/O防熔断节流版)...")
    print(f"[*] 警告: 仅读取元数据，不加载物理像素。\n")

    processed_patients = set()
    stats = {
        'gender': Counter(),
        'age': Counter(),
        'image_size': Counter(),
        'tracer': Counter(),
        'manufacturer': Counter()
    }

    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"[!] 警告: 物理路径不存在，已跳过 -> {root_dir}")
            continue

        print(f"[*] 正在接管总干: {root_dir}")
        for dirpath, _, filenames in os.walk(root_dir):
            dir_lower = dirpath.lower()
            if not ('full' in dir_lower or 'normal' in dir_lower):
                continue

            for f in filenames:
                if f.lower().endswith(('.dcm', '.ima')):
                    file_path = os.path.join(dirpath, f)
                    try:
                        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)

                        # [逻辑修复]：强行读取全球唯一的 StudyInstanceUID，废弃 PatientID
                        study_uid = getattr(dcm, 'StudyInstanceUID', None)
                        if not study_uid:
                            study_uid = dirpath  # 极端情况降级为目录路径防重

                        if study_uid in processed_patients:
                            break  # 该序列已记录，跳过

                        processed_patients.add(study_uid)

                        # [物理修复]：I/O 节流阀，休眠 10 毫秒，防止 VHD 队列溢出熔断
                        time.sleep(0.01)

                        # 终端动态进度刷新
                        sys.stdout.write(f"\r    -> 物理扫描中... 已锚定有效序列: {len(processed_patients):>4} 例 ")
                        sys.stdout.flush()

                        gender = getattr(dcm, 'PatientSex', 'Unknown')
                        age_raw = getattr(dcm, 'PatientAge', 'Unknown')
                        age_group = f"{age_raw[:2]}0s" if age_raw != 'Unknown' and len(age_raw) >= 2 else 'Unknown'

                        rows = getattr(dcm, 'Rows', 'Unknown')
                        cols = getattr(dcm, 'Columns', 'Unknown')
                        size = f"{rows}x{cols}"

                        manufacturer = getattr(dcm, 'Manufacturer', 'Unknown')

                        tracer = 'Unknown'
                        if 'RadiopharmaceuticalInformationSequence' in dcm:
                            seq = dcm.RadiopharmaceuticalInformationSequence[0]
                            tracer = getattr(seq, 'Radiopharmaceutical', 'Unknown')

                        stats['gender'][gender] += 1
                        stats['age'][age_group] += 1
                        stats['image_size'][size] += 1
                        stats['tracer'][tracer] += 1
                        stats['manufacturer'][manufacturer] += 1

                    except Exception as e:
                        pass
                    break

        print()

    print("\n" + "=" * 50)
    print(f"  UDPET 物理底层分布报告 (总探明 Full Dose 序列数: {len(processed_patients)})")
    print("=" * 50)

    for category, counter in stats.items():
        print(f"\n[+] {category.upper()} 分布:")
        for key, val in counter.most_common():
            percentage = (val / len(processed_patients)) * 100 if processed_patients else 0
            print(f"    - {key:<20}: {val:>5} 例 ({percentage:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    TARGET_DIRS = [
        "H:/Bern-Inselspital-2022",
        "H:/Shanghai-Ruijin-Hospital-2022",
        "H:/Shanghai-Ruijin-Hospital-2023"
    ]
    fast_dicom_scanner(TARGET_DIRS)
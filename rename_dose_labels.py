import os
import numpy as np
import glob
import pandas as pd


def rename_and_annotate_dose():
    test_data_folder = "./processed_data_3d_osem/test"
    vis_folder = "./test_visualizations"
    csv_path = os.path.join(vis_folder, "test_metrics_summary.csv")

    if not os.path.exists(test_data_folder) or not os.path.exists(vis_folder):
        print("[!] 找不到数据或可视化文件夹，请检查路径。")
        return

    npy_files = [f for f in os.listdir(test_data_folder) if f.endswith('.npy')]
    print(f"[*] 共发现 {len(npy_files)} 个测试集病例，开始提取 DRF 物理剂量标签...")

    drf_mapping = {}

    # 1. 提取 DRF 并重命名 PNG 图片
    renamed_count = 0
    for npy_file in npy_files:
        npy_path = os.path.join(test_data_folder, npy_file)
        base_name = npy_file.replace('.npy', '')

        try:
            # 仅提取字典元数据，不加载全量张量以节省内存
            data = np.load(npy_path, allow_pickle=True).item()
            drf = data['metadata'].get('drf', -1.0)
            drf_mapping[base_name] = drf
        except Exception as e:
            print(f"[!] 读取 {npy_file} 失败: {e}")
            continue

        # 匹配该病例的所有切片可视化图片
        search_pattern = os.path.join(vis_folder, f"{base_name}_Z*.png")
        png_files = glob.glob(search_pattern)

        for png_path in png_files:
            if "_DRF" in png_path:
                continue  # 拦截已重命名的文件，防止二次污染

            dir_name = os.path.dirname(png_path)
            old_filename = os.path.basename(png_path)
            name_part, ext = os.path.splitext(old_filename)

            # 插入 DRF 标签 (保留 1 位小数)
            new_filename = f"{name_part}_DRF{drf:.1f}{ext}"
            new_png_path = os.path.join(dir_name, new_filename)

            os.rename(png_path, new_png_path)
            renamed_count += 1

    print(f"[*] 成功为 {renamed_count} 张可视化图片追加了剂量标签。")

    # 2. 同步更新 CSV 物理报表
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)

            if 'DRF_Dose' not in df.columns:
                # 依据 File_Name 列映射对应的 DRF 剂量
                df.insert(3, 'DRF_Dose',
                          df['File_Name'].map(drf_mapping).apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "Unknown"))

                # 强行修正 Sample_ID，使其与重命名后的 PNG 文件名保持绝对一致
                df['Sample_ID'] = df.apply(lambda row: f"{row['Sample_ID']}_DRF{row['DRF_Dose']}", axis=1)

                df.to_csv(csv_path, index=False)
                print("[*] 成功在 CSV 报表中追加了 'DRF_Dose' 列，并完成了 Sample_ID 的对齐。")
            else:
                print("[*] CSV 报表中已存在 'DRF_Dose' 列，跳过写入。")
        except Exception as e:
            print(f"[!] 更新 CSV 时引发异常: {e}")
    else:
        print("[!] 未找到 CSV 报表，仅执行了图片重命名。")


if __name__ == "__main__":
    rename_and_annotate_dose()
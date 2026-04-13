import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np
import warnings

# 抑制部分SciPy内部因完全相同序列触发的无用警告
warnings.filterwarnings('ignore')

# 读取生成的三个 CSV 文件
f1 = "./test_visualizations_ud_test_Label_0.100/ud_metrics_summary_test_Label_0.100.csv"
f2 = "./test_visualizations_ud_test_Label_0.250/ud_metrics_summary_test_Label_0.250.csv"
f3 = "./test_visualizations_ud_test_Label_0.500/ud_metrics_summary_test_Label_0.500.csv"

try:
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)
except FileNotFoundError as e:
    print(f"执行中断：找不到对应文件，请确认当前目录存在对应 CSV。报错信息: {e}")
    exit()

# 汇总合并数据
df = pd.concat([df1, df2, df3], ignore_index=True)

# 强制转换数据类型
metrics = ['PSNR_Out', 'SSIM_Out', 'Bias_Out(%)']
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors='coerce')

df['Physical_Dose'] = df['Physical_Dose'].astype(float).round(3).astype(str)
df['Inference_Label'] = df['Inference_Label'].astype(float).round(3).astype(str)

dose_order = sorted(df['Physical_Dose'].unique(), key=float)
label_order = sorted(df['Inference_Label'].unique(), key=float)

# 预先计算所有差值与统计学结果
diff_data = []
stats_annotations = {}

for metric in metrics:
    for dose in dose_order:
        df_dose = df[df['Physical_Dose'] == dose]

        # --- 门控机制：Friedman 全局检验 ---
        df_pivot = df_dose.pivot(index='Sample_ID', columns='Inference_Label', values=metric).dropna()
        global_sig = False
        if len(df_pivot) >= 5:
            try:
                stat_f, p_f = friedmanchisquare(*[df_pivot[lbl] for lbl in label_order])
                global_sig = (p_f < 0.05)
            except ValueError:
                global_sig = False

        df_matched = df_dose[df_dose['Inference_Label'] == dose].set_index('Sample_ID')

        for mis_label in label_order:
            if mis_label == dose:
                continue

            df_mis = df_dose[df_dose['Inference_Label'] == mis_label].set_index('Sample_ID')
            common_ids = df_matched.index.intersection(df_mis.index)

            if len(common_ids) < 5:
                continue

            delta_series = df_mis.loc[common_ids, metric] - df_matched.loc[common_ids, metric]

            for sid, val in delta_series.items():
                diff_data.append({
                    'Metric': metric,
                    'Physical_Dose': dose,
                    'Mismatch_Label': mis_label,
                    'Delta': val
                })

            # --- 事后 Wilcoxon 配对检验与 Bonferroni 多重比较校正 ---
            if not global_sig:
                p_adj = 1.0
            else:
                try:
                    stat, p_val = wilcoxon(df_matched.loc[common_ids, metric], df_mis.loc[common_ids, metric])
                    p_adj = min(p_val * 2.0, 1.0)
                except ValueError:
                    p_adj = 1.0

            if p_adj < 0.001:
                sig = "***"
            elif p_adj < 0.01:
                sig = "**"
            elif p_adj < 0.05:
                sig = "*"
            else:
                sig = "ns"

            stats_annotations[(metric, dose, mis_label)] = sig

df_diff = pd.DataFrame(diff_data)

# ================= 绘图阶段 =================
sns.set_theme(style="whitegrid")

# 创建 3x3 的分面网格
fig, axes = plt.subplots(3, 3, figsize=(16, 15))

# 视觉参数设定
meanprops = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "5"}
flierprops = {"marker": "o", "markersize": 3, "alpha": 0.4, "markeredgecolor": "none", "markerfacecolor": "gray"}

# 固定颜色映射：确保同一个推断标签在所有图表中颜色一致
color_palette = sns.color_palette("Set2", len(label_order))
palette_dict = {lbl: col for lbl, col in zip(label_order, color_palette)}

for i, metric in enumerate(metrics):
    # 获取该指标全局的最值，以统一整行的 Y 轴范围
    metric_diffs = df_diff[df_diff['Metric'] == metric]['Delta']
    if metric_diffs.empty: continue

    y_min, y_max = metric_diffs.min(), metric_diffs.max()
    y_range = y_max - y_min
    y_lim_bottom = y_min - y_range * 0.1
    y_lim_top = y_max + y_range * 0.15

    for j, dose in enumerate(dose_order):
        ax = axes[i, j]
        plot_data = df_diff[(df_diff['Metric'] == metric) & (df_diff['Physical_Dose'] == dose)]

        if plot_data.empty:
            ax.set_visible(False)
            continue

        # 仅包含错配标签
        mis_order = [lbl for lbl in label_order if lbl != dose]

        sns.boxplot(
            data=plot_data, x='Mismatch_Label', y='Delta',
            order=mis_order, ax=ax, width=0.5,
            showmeans=True, meanprops=meanprops, palette=palette_dict,
            whis=2.5, flierprops=flierprops
        )

        # 添加 Y=0 红色基准线
        ax.axhline(0, color='red', linestyle='--', linewidth=2.0, alpha=1.0, zorder=0)
        ax.set_ylim(y_lim_bottom, y_lim_top)

        # 标题与坐标轴标签处理
        if i == 0:
            ax.set_title(f'True Physical Dose: {dose}', fontsize=14, fontweight='bold')

        if j == 0:
            ax.set_ylabel(f'Δ {metric}\n(Mismatch - Match)', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')

        ax.set_xlabel('Mismatch Label', fontsize=12)

        # 添加统计学星号标注
        for x_idx, mis_label in enumerate(mis_order):
            sig = stats_annotations.get((metric, dose, mis_label), "ns")
            box_data = plot_data[plot_data['Mismatch_Label'] == mis_label]['Delta']

            if not box_data.empty:
                text_y = box_data.max() + y_range * 0.03
                ax.text(x_idx, text_y, sig, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

plt.tight_layout()
output_img = 'paired_difference_boxplots.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"网格化分面图表绘制完毕，已保存为 {output_img}。")
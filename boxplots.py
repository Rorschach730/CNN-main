import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np
import warnings

# 抑制部分SciPy内部因完全相同序列触发的无用警告
warnings.filterwarnings('ignore')

# 读取生成的三个 CSV 文件 (请确保路径与你本地一致)
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

# 强制从小到大排序
dose_order = sorted(df['Physical_Dose'].unique(), key=float)
label_order = sorted(df['Inference_Label'].unique(), key=float)

# 设置基础画布与样式
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# 定义平均值的标记样式
meanprops = {
    "marker": "o", "markerfacecolor": "white",
    "markeredgecolor": "black", "markersize": "5"
}

# 方案A：定义离散点（Outliers）的视觉样式：缩小尺寸并加半透明灰色
flierprops = {
    "marker": "o",
    "markersize": 3,
    "alpha": 0.4,
    "markeredgecolor": "none",
    "markerfacecolor": "gray"
}

# 计算 Seaborn boxplot 中不同 hue 对应的 X 轴偏移量
n_hues = len(label_order)
width = 0.8
offsets = np.linspace(0, width - width / n_hues, n_hues)
offsets -= offsets.mean()

# 遍历绘制三个核心指标的箱型图
for i, metric in enumerate(metrics):
    ax = axes[i]

    # 保留已确认的 whis=2.5 及 flierprops 设置
    sns.boxplot(
        data=df, x='Physical_Dose', y=metric, hue='Inference_Label',
        order=dose_order, hue_order=label_order,
        ax=ax, showmeans=True, meanprops=meanprops, palette="Set2",
        whis=2.5, flierprops=flierprops
    )

    ax.set_title(f'{metric} (Grouped by Physical Dose)', fontsize=14, fontweight='bold')
    ax.set_xlabel('True Physical Dose', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)

    if i == 0:
        ax.legend(title='Inference Label', title_fontsize='11', fontsize='10')
    else:
        ax.get_legend().remove()

    y_range = df[metric].max() - df[metric].min()

    # 全局记录当前子图面板中连线所抵达的最高Y坐标
    max_line_y = df[metric].max()

    for x_idx, dose in enumerate(dose_order):
        match_idx = label_order.index(dose)
        df_dose = df[df['Physical_Dose'] == dose]

        # --- 门控机制：Friedman 全局检验 ---
        # 确保提取所有标签下同一个 Sample_ID 的配对数据
        df_pivot = df_dose.pivot(index='Sample_ID', columns='Inference_Label', values=metric).dropna()

        global_sig = False
        if len(df_pivot) >= 5:
            try:
                # 对当前剂量下所有组别进行非参数重复测量方差分析
                stat_f, p_f = friedmanchisquare(*[df_pivot[lbl] for lbl in label_order])
                global_sig = (p_f < 0.05)
            except ValueError:
                global_sig = False

        df_matched = df_dose[df_dose['Inference_Label'] == dose].sort_values('Sample_ID')

        step_h = y_range * 0.03
        current_y = df_dose[metric].max() + step_h

        for mis_idx, mis_label in enumerate(label_order):
            if mis_idx == match_idx:
                continue

            df_mis = df_dose[df_dose['Inference_Label'] == mis_label].sort_values('Sample_ID')
            merged = pd.merge(df_matched, df_mis, on='Sample_ID', suffixes=('_m', '_mis'))

            if len(merged) < 5:
                continue

            # --- 事后检验与多重比较校正 ---
            if not global_sig:
                # 全局检验未通过，阻断后续检验，直接判定为 ns
                p_adj = 1.0
            else:
                try:
                    stat, p_val = wilcoxon(merged[f'{metric}_m'], merged[f'{metric}_mis'])
                    # Bonferroni 校正：因与两个错配组比较，比较次数 m=2，调整 p 值
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

            x1 = x_idx + offsets[match_idx]
            x2 = x_idx + offsets[mis_idx]

            ax.plot([x1, x1, x2, x2], [current_y, current_y + step_h, current_y + step_h, current_y], lw=1.2, c='black')
            ax.text((x1 + x2) * 0.5, current_y + step_h, sig, ha='center', va='bottom', color='black', fontsize=12)

            current_y += step_h * 3.5

        # 内层连线画完后，将全局最大高度与当前剂量高度对比取最大值
        max_line_y = max(max_line_y, current_y)

    # 动态调高 Y 轴上限
    ax.set_ylim(df[metric].min() - y_range * 0.05, max_line_y + y_range * 0.1)

plt.tight_layout()

output_img = 'mismatch_boxplots.png'
plt.savefig(output_img, dpi=300)
print(f"数据可视化执行完毕，带有严谨门控与多重比较校正的图表已保存为 {output_img}。")
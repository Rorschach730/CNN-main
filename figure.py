import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl

# 设置全局字体为顶刊通用的 Arial/Helvetica
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']


def plot_top_journal_boxplots(csv_path):
    df = pd.read_csv(csv_path)

    # 指标配对与显示名称
    metric_pairs = [
        ('ROI_PSNR_In', 'ROI_PSNR_Out', 'PSNR (dB)'),
        ('ROI_SSIM_In', 'ROI_SSIM_Out', 'SSIM'),
        ('ROI_SNR_In', 'ROI_SNR_Out', 'SNR (dB)'),
        ('ROI_Bias_In(%)', 'ROI_Bias_Out(%)', 'Bias (%)'),
        ('ROI_MAPE_In(%)', 'ROI_MAPE_Out(%)', 'MAPE (%)'),
        ('ROI_CR_In(%)', 'ROI_CR_Out(%)', 'CR (%)')
    ]

    # 设置学术配色 (Nature/Science 风格)
    palette = ["#4C72B0", "#DD8452"]  # 经典深蓝与砖橙

    # 创建 2x3 画布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes = axes.flatten()

    for i, (in_col, out_col, name) in enumerate(metric_pairs):
        ax = axes[i]

        # 准备数据
        plot_data = pd.DataFrame({
            'Method': ['MLEM'] * len(df) + ['JiT'] * len(df),
            'Value': pd.concat([df[in_col], df[out_col]])
        })

        # 绘制箱线图：隐藏离群点 (showfliers=False) 以保持画面整洁
        sns.boxplot(x='Method', y='Value', data=plot_data, ax=ax,
                    palette=palette, width=0.5, showfliers=False,
                    linewidth=1.2)

        # Wilcoxon 显著性检验
        _, p = stats.wilcoxon(df[in_col], df[out_col])
        star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

        # 绘制显著性标注连线
        # 获取当前 y 轴范围
        y_min, y_max = ax.get_ylim()
        h_range = y_max - y_min
        line_y = y_max + h_range * 0.02
        line_h = h_range * 0.02

        # 连线坐标
        ax.plot([0, 0, 1, 1], [line_y, line_y + line_h, line_y + line_h, line_y],
                lw=1.0, c='black')
        ax.text(0.5, line_y + line_h, star, ha='center', va='bottom',
                color='black', fontsize=14, fontweight='bold')

        # 美化坐标轴
        ax.set_title(name, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('')  # 移除子图横坐标标签
        ax.set_ylabel('')
        ax.set_xticklabels([])  # 彻底移除子图刻度标签，由全局图例接管
        ax.tick_params(axis='y', labelsize=12)
        sns.despine(ax=ax)  # 移除上方和右侧的边框，这是顶刊的标准操作

    # 关键：创建一个全局共享图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette]
    labels = ['Standard MLEM (Input)', 'JiT-Denoised (Output)']
    fig.legend(handles, labels, loc='upper center', ncol=2,
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.98))

    # 调整布局，为顶部的图例留出空间
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Publication_Boxplot_Final.png")
    plt.show()


# 运行
plot_top_journal_boxplots("ud_clinical_metrics.csv")
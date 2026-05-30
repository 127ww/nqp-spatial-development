# -*- coding: utf-8 -*-
"""
图7：2011年与2023年中国各省新质生产力 & 高质量发展莫兰散点图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Arial", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

SCATTER   = '#0F4D92'
FIT_LINE  = '#B64342'
GRID      = '#E0E0E0'
BG        = '#FFFFFF'

def clean_name(name):
    return str(name).replace('省', '').replace('市', '').replace('自治区', '')\
                     .replace('维吾尔', '').replace('回族', '').replace('壮族', '')

# 读取数据
try:
    df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='gbk')
except:
    df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='utf-8-sig')
df['省份'] = df['省份'].apply(clean_name)

w_df = pd.read_csv('data/W_adj_norm.csv', header=None)
STANDARD_PROVINCES = [
    '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
    '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
    '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
]
w_df.index = STANDARD_PROVINCES
w_df.columns = STANDARD_PROVINCES
W_norm = w_df.values.astype(float)

VARIABLES = {'新质生产力_组合得分': '新质生产力', '高质量发展_组合得分': '高质量发展'}
YEARS = [2011, 2023]

for var_col, var_name in VARIABLES.items():
    prefix = "XZ" if "新质" in var_name else "GZ"
    for year in YEARS:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
        ax.set_facecolor(BG)

        df_year = df[df['年份'] == year].set_index('省份').reindex(STANDARD_PROVINCES)
        x = df_year[var_col].values
        z = (x - np.mean(x)) / np.std(x)
        wz = np.dot(W_norm, z)

        slope, intercept, r_value, p_value, std_err = stats.linregress(z, wz)

        ax.grid(True, color=GRID, linestyle='-', linewidth=0.5, alpha=0.8, zorder=1)
        ax.axhline(0, color='#333333', linewidth=0.8, zorder=2)
        ax.axvline(0, color='#333333', linewidth=0.8, zorder=2)
        ax.scatter(z, wz, color=SCATTER, s=70, edgecolors='white', linewidths=0.8, zorder=4)

        for j, txt in enumerate(STANDARD_PROVINCES):
            ax.annotate(txt, (z[j], wz[j]), xytext=(4, 3), textcoords='offset points',
                       fontsize=9, color='#333333')

        line_x = np.array([-3.5, 4.5])
        line_y = intercept + slope * line_x
        ax.plot(line_x, line_y, color=FIT_LINE, linewidth=2.2, zorder=3,
                label=f"Moran's I = {slope:.4f}  (p = {p_value:.4f})")

        ax.set_title(f'{var_name}莫兰散点图（{year}年）', fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('标准化得分 (z)', fontsize=12)
        ax.set_ylabel('空间滞后 (Wz)', fontsize=12)
        ax.set_xlim(-3.5, 4.5)
        ax.set_ylim(-3.5, 3.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        q_style = dict(fontsize=11, color='#999999', alpha=0.7, ha='center', fontweight='bold')
        ax.text(3.0, 3.0, 'H-H', **q_style)
        ax.text(-2.5, 3.0, 'L-H', **q_style)
        ax.text(-2.5, -2.8, 'L-L', **q_style)
        ax.text(3.0, -2.8, 'H-L', **q_style)

        ax.legend(loc='lower right', fontsize=11, frameon=True, edgecolor=GRID)

        plt.tight_layout()

        def save_pub_py(fig, name, dpi=600):
            fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
            fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
            fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

        save_pub_py(fig, f'figures/Moran_{prefix}_{year}')
        plt.close(fig)

print("图7已保存")

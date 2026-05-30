# -*- coding: utf-8 -*-
"""
图6：2023年新质生产力区域分布差异箱线图 + 省际排名条形图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

EAST   = "#B64342"
MIDWEST = "#0F4D92"
GRID   = "#E0E0E0"
BG     = "#FFFFFF"
GREY   = "#767676"

df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='gbk', engine='python')
latest_year = df['年份'].max()
df_latest = df[df['年份'] == latest_year].copy()

east_list = ['北京', '天津', '河北', '辽宁', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南']
df_latest['区域'] = df_latest['省份'].apply(lambda x: '东部地区' if x in east_list else '中西部地区')

region_palette = {'东部地区': EAST, '中西部地区': MIDWEST}

# ---- 箱线图 + 散点叠加 ----
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(BG)

sns.boxplot(x='区域', y='新质生产力_组合得分', hue='区域', data=df_latest,
            palette=region_palette, width=0.4, linewidth=1.2, legend=False,
            flierprops=dict(marker='o', markersize=5, alpha=0.4),
            ax=ax)
sns.stripplot(x='区域', y='新质生产力_组合得分', data=df_latest,
              color='#333333', alpha=0.45, jitter=True, size=6, ax=ax)

ax.set_title(f'{latest_year}年新质生产力区域分布差异', fontsize=15, fontweight='bold', pad=12)
ax.set_ylabel('新质生产力综合得分', fontsize=12)
ax.set_xlabel('')
ax.grid(axis='y', color=GRID, linestyle='-', linewidth=0.5, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "figures/boxplot_distribution")
plt.close()

# ---- 省际排名条形图 ----
df_sorted = df_latest.sort_values(by='新质生产力_组合得分', ascending=True)
colors_bar = [EAST if p in east_list else MIDWEST for p in df_sorted['省份']]

fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
ax.set_facecolor(BG)

ax.barh(df_sorted['省份'], df_sorted['新质生产力_组合得分'], color=colors_bar,
        edgecolor='white', linewidth=0.5, height=0.7)
ax.set_title(f'{latest_year}年中国各省份新质生产力得分排名', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('新质生产力综合得分', fontsize=12)
ax.grid(axis='x', color=GRID, linestyle='-', linewidth=0.5, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=EAST, label='东部地区'),
                   Patch(facecolor=MIDWEST, label='中西部地区')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True)

plt.tight_layout()
save_pub_py(fig, "figures/sorted_bar_chart")
plt.close()

print("图6已保存: boxplot_distribution.png + sorted_bar_chart.png")

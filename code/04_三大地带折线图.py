# -*- coding: utf-8 -*-
"""
图4：2011-2023年中国三大地带新质生产力演进趋势
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

EAST     = "#B64342"
CENTRAL  = "#0F4D92"
WEST     = "#2E9E44"
NATIONAL = "#767676"
GRID = "#E0E0E0"
BG = "#FFFFFF"

# 区域定义
EAST_LIST = ['北京', '天津', '河北', '辽宁', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南']
CENTRAL_LIST = ['山西', '吉林', '黑龙江', '安徽', '江西', '河南', '湖北', '湖南']
WEST_LIST = ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']

def get_region(p):
    if p in EAST_LIST: return '东部地区'
    elif p in CENTRAL_LIST: return '中部地区'
    elif p in WEST_LIST: return '西部地区'
    return '未知'

df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='gbk', engine='python')
df['区域'] = df['省份'].apply(get_region)

region_trend = df.groupby(['年份', '区域'])['新质生产力_组合得分'].mean().unstack()
national_trend = df.groupby('年份')['新质生产力_组合得分'].mean()

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.set_facecolor(BG)

ax.plot(region_trend.index, region_trend['东部地区'], marker='o', markersize=7,
        linewidth=2.5, color=EAST, label='东部地区', zorder=4)
ax.plot(national_trend.index, national_trend, marker='s', markersize=6,
        linewidth=2.2, color=NATIONAL, linestyle='--', label='全国平均', zorder=3)
ax.plot(region_trend.index, region_trend['中部地区'], marker='^', markersize=7,
        linewidth=2.2, color=CENTRAL, label='中部地区', zorder=3)
ax.plot(region_trend.index, region_trend['西部地区'], marker='D', markersize=6,
        linewidth=2.2, color=WEST, label='西部地区', zorder=3)

ax.set_title('2011—2023年中国三大地带新质生产力演进趋势', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('新质生产力均值得分', fontsize=12)
ax.set_xticks(region_trend.index)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', color=GRID, linestyle='-', linewidth=0.5, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor=GRID)

plt.tight_layout()

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "figures/三大地带新质生产力演进折线图")
plt.close()
print("图4已保存")

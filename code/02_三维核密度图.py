# -*- coding: utf-8 -*-
"""
图5：新质生产力三维核密度演进图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.collections import PolyCollection

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

COLORS = ['#0F4D92', '#7EB5D6', '#E9A6A1', '#B64342']  # 蓝→红渐变
YEARS  = [2011, 2015, 2019, 2023]

df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='gbk', engine='python')

fig = plt.figure(figsize=(10, 7), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

verts = []
max_density = 0

for i, year in enumerate(YEARS):
    data = df[df['年份'] == year]['新质生产力_组合得分'].dropna().values
    kde = gaussian_kde(data, bw_method=0.3)
    x = np.linspace(0, 0.8, 200)
    y = kde(x)
    if max(y) > max_density:
        max_density = max(y)

    x_poly = np.concatenate([[x[0]], x, [x[-1]]])
    y_poly = np.concatenate([[0], y, [0]])
    verts.append(list(zip(x_poly, y_poly)))

    ax.plot(x, [year] * len(x), y, color=COLORS[i], linewidth=2.5)

poly = PolyCollection(verts, facecolors=COLORS, alpha=0.55, edgecolors='none')
ax.add_collection3d(poly, zs=YEARS, zdir='y')

ax.set_xlabel('\n新质生产力综合得分', fontsize=12, labelpad=10)
ax.set_ylabel('\n年份', fontsize=12, labelpad=10)
ax.set_zlabel('\n核密度值', fontsize=12, labelpad=10)

ax.set_xlim(0, 0.8)
ax.set_ylim(2010, 2024)
ax.set_zlim(0, max_density * 1.1)
ax.set_yticks(YEARS)
ax.view_init(elev=25, azim=-50)

# 图例
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=c, linewidth=2.5, label=str(y))
                   for c, y in zip(COLORS, YEARS)]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "figures/新质生产力_三维核密度演进图")
plt.close()
print("图5已保存")

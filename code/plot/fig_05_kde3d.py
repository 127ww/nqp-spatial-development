# -*- coding: utf-8 -*-
"""Figure 5: 3D kernel density — widening distribution over time."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.collections import PolyCollection
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Arial", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
})

df = pd.read_csv("data/面板数据_最终测度得分结果.csv", encoding="gbk")

# Blue-to-red temporal gradient (nature palette family)
COLORS = ["#0F4D92", "#3775BA", "#E9A6A1", "#B64342"]
YEARS  = [2011, 2015, 2019, 2023]

fig = plt.figure(figsize=(8, 5.5), facecolor="white")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

verts = []
max_density = 0

for i, year in enumerate(YEARS):
    data = df[df["年份"] == year]["新质生产力_组合得分"].dropna().values
    kde = gaussian_kde(data, bw_method=0.3)
    x = np.linspace(0, 0.75, 200)
    y = kde(x)
    max_density = max(max_density, y.max())

    x_poly = np.concatenate([[x[0]], x, [x[-1]]])
    y_poly = np.concatenate([[0], y, [0]])
    verts.append(list(zip(x_poly, y_poly)))

    ax.plot(x, [year] * len(x), y, color=COLORS[i], lw=2.0)

poly = PolyCollection(verts, facecolors=COLORS, alpha=0.45, edgecolors="none")
ax.add_collection3d(poly, zs=YEARS, zdir="y")

ax.set_xlabel("新质生产力得分", fontsize=9, labelpad=8)
ax.set_ylabel("年份", fontsize=9, labelpad=8)
ax.set_zlabel("核密度", fontsize=9, labelpad=8)

ax.set_xlim(0, 0.75)
ax.set_ylim(2010, 2024)
ax.set_zlim(0, max_density * 1.1)
ax.set_yticks(YEARS)
ax.view_init(elev=22, azim=-55)

# Direct year labels
from matplotlib.lines import Line2D
for c, y in zip(COLORS, YEARS):
    ax.text2D(0.02, 0.94 - YEARS.index(y)*0.06,
              str(y), transform=ax.transAxes, color=c,
              fontsize=9, fontweight="bold")

ax.set_title("新质生产力分布演进：波峰右移、分布拓宽",
             fontsize=11, fontweight="bold", color="#272727", pad=16)

plt.tight_layout(pad=1.5)

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_05_kde3d")
plt.close()
print("Figure 5 saved.")

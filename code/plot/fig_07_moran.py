# -*- coding: utf-8 -*-
"""Figure 7: Moran scatter — persistent spatial autocorrelation."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
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

BLUE  = "#0F4D92"
RED   = "#B64342"
GREY  = "#767676"
DARK  = "#272727"

def clean(name):
    return str(name).replace("省","").replace("市","").replace("自治区","")\
                     .replace("维吾尔","").replace("回族","").replace("壮族","")

df = pd.read_csv("data/面板数据_最终测度得分结果.csv", encoding="gbk")
df["省份"] = df["省份"].apply(clean)

W = pd.read_csv("data/W_adj_norm.csv", header=None).values
STANDARD = ["北京","天津","河北","山西","内蒙古","辽宁","吉林","黑龙江",
            "上海","江苏","浙江","安徽","福建","江西","山东","河南",
            "湖北","湖南","广东","广西","海南","重庆","四川","贵州",
            "云南","西藏","陕西","甘肃","青海","宁夏","新疆"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white")

for ax, year in zip(axes, [2011, 2023]):
    ax.set_facecolor("white")
    d = df[df["年份"] == year].set_index("省份").reindex(STANDARD)
    x = d["新质生产力_组合得分"].values
    z = (x - x.mean()) / x.std()
    wz = W.dot(z)

    slope, intercept, r, p, _ = stats.linregress(z, wz)

    ax.axhline(0, color=GREY, lw=0.5, zorder=1)
    ax.axvline(0, color=GREY, lw=0.5, zorder=1)

    colors = [RED if s in ["北京","上海","江苏","浙江","广东","天津","福建","山东"]
              else (BLUE if wz_i > slope * z_i + intercept else GREY)
              for s, z_i, wz_i in zip(STANDARD, z, wz)]
    sizes  = [55 if s in ["北京","上海","广东","江苏","浙江"] else 35 for s in STANDARD]

    ax.scatter(z, wz, c=colors, s=sizes, edgecolors="white", lw=0.5, zorder=4)

    line_x = np.array([-3.5, 4.5])
    ax.plot(line_x, intercept + slope * line_x, color=RED, lw=1.5, zorder=3)

    # Label key provinces only
    key = ["北京","上海","广东","江苏","浙江","西藏","青海","甘肃","贵州"]
    for j, name in enumerate(STANDARD):
        if name in key:
            ax.annotate(name, (z[j], wz[j]), xytext=(5, 4), textcoords="offset points",
                       fontsize=6, color=DARK)

    # Quadrant labels
    ax.text(3.0, 3.0, "HH", ha="center", fontsize=7, color=GREY, fontweight="bold")
    ax.text(-2.5, 3.0, "LH", ha="center", fontsize=7, color=GREY, fontweight="bold")
    ax.text(-2.5, -2.8, "LL", ha="center", fontsize=7, color=GREY, fontweight="bold")
    ax.text(3.0, -2.8, "HL", ha="center", fontsize=7, color=GREY, fontweight="bold")

    ax.set_title("{} 年".format(year), fontsize=9, fontweight="bold", color=DARK)
    ax.set_xlabel("标准化得分 (z)", fontsize=8)
    ax.set_ylabel("空间滞后 (Wz)", fontsize=8)
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-3.5, 3.5)

    # Moran's I in corner
    ax.text(0.03, 0.97, "Moran's I = {:.3f}\np = {:.4f}".format(slope, p),
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.85))

fig.suptitle("新质生产力空间自相关：显著的HH-LL集聚与路径依赖",
             fontsize=11, fontweight="bold", color=DARK, y=1.02)

plt.tight_layout(pad=2.0)

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_07_moran")
plt.close()
print("Figure 7 saved.")

# -*- coding: utf-8 -*-
"""Figure 4: Three-region NQP trend — rising divergence."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── CJK-first (user memory) ───────────────────────────────
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

# ── Data ──────────────────────────────────────────────────
df = pd.read_csv("data/面板数据_最终测度得分结果.csv", encoding="gbk")

east = ["北京","天津","河北","辽宁","上海","江苏","浙江","福建","山东","广东","海南"]
central = ["山西","吉林","黑龙江","安徽","江西","河南","湖北","湖南"]
west = ["内蒙古","广西","重庆","四川","贵州","云南","西藏","陕西","甘肃","青海","宁夏","新疆"]

def region(p):
    if p in east: return "东部"
    if p in central: return "中部"
    return "西部"

df["区域"] = df["省份"].apply(region)
pivot = df.groupby(["年份","区域"])["新质生产力_组合得分"].mean().unstack()
national = df.groupby("年份")["新质生产力_组合得分"].mean()

years = pivot.index.values

# ── Palette: one blue family + neutral ────────────────────
C = {"东部": "#0F4D92", "中部": "#3775BA", "西部": "#7EB5D6", "全国": "#767676"}

# ── Build ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
ax.set_facecolor("white")

for region_name in ["东部", "中部", "西部"]:
    ax.plot(years, pivot[region_name], color=C[region_name], lw=2.0,
            marker="o", ms=4.5, mec="white", mew=0.6)
ax.plot(years, national, color=C["全国"], lw=1.2, ls=(0, (4, 2)), marker="s",
        ms=4, mec="white", mew=0.4)

# ── Direct labels at line ends ────────────────────────────
x_end = years[-1]
for name in ["东部", "中部", "西部", "全国"]:
    y_end = pivot[name].iloc[-1] if name != "全国" else national.iloc[-1]
    ax.text(x_end + 0.15, y_end, name, color=C[name],
            fontsize=8, fontweight="bold", va="center", ha="left")

# ── Annotate gap ──────────────────────────────────────────
gap_start = pivot["东部"].iloc[0] - pivot["西部"].iloc[0]
gap_end   = pivot["东部"].iloc[-1] - pivot["西部"].iloc[-1]
ax.text(0.03, 0.94,
        "东西差距: {:.3f} → {:.3f}".format(gap_start, gap_end),
        transform=ax.transAxes,
        fontsize=7, color="#B64342", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#E9A6A1", alpha=0.85, lw=0.5))

# ── Axis ──────────────────────────────────────────────────
ax.set_xlim(2010.5, 2024.5)
ax.set_xticks([2011, 2013, 2015, 2017, 2019, 2021, 2023])
ax.set_xticklabels([str(y) for y in [2011,2013,2015,2017,2019,2021,2023]])
ax.set_ylabel("新质生产力均值得分", fontsize=9, labelpad=4)
ax.spines["left"].set_position(("outward", 6))
ax.spines["bottom"].set_position(("outward", 6))

# ── Title ─────────────────────────────────────────────────
ax.set_title("新质生产力区域演进：稳步跃升与空间极化并存",
             fontsize=11, fontweight="bold", color="#272727", pad=10)

# ── Footer ────────────────────────────────────────────────
fig.text(0.5, 0.01, "n = 31 省份, 2011–2023。东部 11 / 中部 8 / 西部 12 省份均值。",
         ha="center", fontsize=6.5, color="#767676")

plt.tight_layout(pad=2.0, rect=[0, 0.04, 1, 1])

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_04_trend")
plt.close()
print("Figure 4 saved.")

# -*- coding: utf-8 -*-
"""Figure 6: Regional boxplot + ranked bar chart of NQP scores."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
latest = df[df["年份"] == df["年份"].max()].copy()

east = ["北京","天津","河北","辽宁","上海","江苏","浙江","福建","山东","广东","海南"]
latest["区域"] = latest["省份"].apply(lambda x: "东部" if x in east else "中西部")

BLUE  = "#0F4D92"
RED   = "#B64342"
GREY  = "#767676"
DARK  = "#272727"

# ── Panel layout ──────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white",
                                  gridspec_kw={"width_ratios": [1, 1.6]})
for ax in [ax_l, ax_r]:
    ax.set_facecolor("white")

# ── Left: boxplot ────────────────────────────────────────
regions = ["东部", "中西部"]
box_data = [latest[latest["区域"] == r]["新质生产力_组合得分"].values for r in regions]

bp = ax_l.boxplot(box_data, labels=regions, widths=0.4,
                  patch_artist=True, medianprops={"color": "white", "lw": 1.2},
                  flierprops={"marker": "o", "ms": 4, "alpha": 0.4, "mec": GREY, "mfc": "none"},
                  whiskerprops={"color": GREY, "lw": 0.8},
                  capprops={"color": GREY, "lw": 0.8})
bp["boxes"][0].set_facecolor(RED)
bp["boxes"][1].set_facecolor(BLUE)

# Overlay individual points
for i, r in enumerate(regions):
    y = box_data[i]
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(y))
    ax_l.scatter(np.full(len(y), i+1) + jitter, y, color=GREY, alpha=0.5, s=18,
                edgecolors="white", lw=0.3, zorder=5)

ax_l.set_ylabel("新质生产力综合得分", fontsize=9)
ax_l.set_title("区域分布差异 (2023)", fontsize=9, fontweight="bold", color=DARK, loc="left")

# ── Right: horizontal bar ────────────────────────────────
latest_sorted = latest.sort_values("新质生产力_组合得分", ascending=True)
colors_bar = [RED if p in east else BLUE for p in latest_sorted["省份"]]

ax_r.barh(latest_sorted["省份"], latest_sorted["新质生产力_组合得分"],
          color=colors_bar, edgecolor="white", lw=0.3, height=0.7)
ax_r.set_title("省际得分排名 (2023)", fontsize=9, fontweight="bold", color=DARK, loc="left")
ax_r.set_xlabel("新质生产力综合得分", fontsize=8)
ax_r.axvline(x=latest_sorted["新质生产力_组合得分"].mean(), color=GREY, lw=0.6, ls=":",
             alpha=0.6)
ax_r.text(latest_sorted["新质生产力_组合得分"].mean() + 0.005, 1.5,
          "均值", fontsize=6.5, color=GREY)

# Legend
from matplotlib.patches import Patch
ax_r.legend(handles=[Patch(facecolor=RED, label="东部"), Patch(facecolor=BLUE, label="中西部")],
            fontsize=7.5, loc="lower right", frameon=True, facecolor="white",
            edgecolor="#DDDDDD", framealpha=0.85)

fig.suptitle("新质生产力空间极化：东部整体领先，中西部低位集中",
             fontsize=11, fontweight="bold", color=DARK, y=1.02)

plt.tight_layout(pad=2.0)

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_06_boxplot")
plt.close()
print("Figure 6 saved.")

# -*- coding: utf-8 -*-
"""Spatial spillover reversal — trickle-down to polarization."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── CJK-safe rcParams (nature-figure spec) ─────────────────
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

# ── Palette ───────────────────────────────────────────────
BLUE = "#0F4D92"
BLUE_LIGHT = "#7EB5D6"
RED = "#B64342"
RED_SOFT = "#E9A6A1"
GREY = "#767676"

# ── Data ──────────────────────────────────────────────────
cats = ["直接效应\n(Direct)", "间接效应\n(Indirect)"]
e_mean = [0.2487, 0.3568]
e_se   = [0.1141, 0.1619]
e_p    = [0.0047, 0.0275]
l_mean = [0.1711, -0.1416]
l_se   = [0.0344, 0.0583]
l_p    = [0.0000, 0.0152]

# ── Build figure ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
ax.set_facecolor("white")

x = np.arange(len(cats))
w = 0.28
gap = 0.05

eb = ax.bar(x - w/2 - gap/2, e_mean, w, color=[BLUE, RED],
            edgecolor="white", linewidth=0.4,
            yerr=e_se, capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": GREY})

lb = ax.bar(x + w/2 + gap/2, l_mean, w, color=[BLUE_LIGHT, RED_SOFT],
            edgecolor="white", linewidth=0.4,
            yerr=l_se, capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": GREY})

ax.axhline(y=0, color=GREY, linewidth=0.5, linestyle=":", alpha=0.4)

# ── Annotate bars ─────────────────────────────────────────
def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""

ax.set_ylim(-0.32, 0.65)

for bars, vals, ses, ps in [(eb, e_mean, e_se, e_p), (lb, l_mean, l_se, l_p)]:
    for b, v, s, p in zip(bars, vals, ses, ps):
        if v >= 0:
            yt = v + s + 0.03
            va = "bottom"
        else:
            yt = v - s - 0.05
            va = "top"
        ax.text(b.get_x() + b.get_width()/2, yt,
                "{:+.4f}{}".format(v, stars(p)),
                ha="center", va=va, fontsize=8, fontweight="bold", color="#272727")

# ── Reversal indicator (巧妙利用右侧负数柱子上方的天然空白) ────────
diff_val = l_mean[1] - e_mean[1]

# x=1.165 刚好是对齐右侧那根粉色(后期)柱子的中心
text_x = 1.165 
text_y = 0.18   

# 绘制悬浮文本框
ax.text(text_x, text_y, 
        f"涓滴退化\nΔ = {diff_val:+.4f}",
        ha="center", va="center", fontsize=8.5, color=RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8F8", edgecolor=RED_SOFT, alpha=1.0),
        zorder=5)


# ── Legend ────────────────────────────────────────────────
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=BLUE, edgecolor="white", label="2011–2017"),
    Patch(facecolor=BLUE_LIGHT, edgecolor="white", label="2018–2023"),
], fontsize=8, loc="upper left", frameon=True,
   facecolor="white", edgecolor="#DDDDDD", framealpha=0.85)

# ── Labels ────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("边际效应估计值", fontsize=9, labelpad=4)
ax.set_yticks([-0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6])

ax.set_title("新质生产力空间溢出的动态演进",
             fontsize=10, fontweight="bold", color="#272727", pad=10)

# ── Footer ────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "注：*** p<0.01, ** p<0.05。误差线=Monte Carlo SE (5000次)。间接效应+为涓滴/−为极化。",
         ha="center", fontsize=6.5, color=GREY)

plt.tight_layout(pad=1.5, rect=[0, 0.04, 1, 1])

def save_pub_py(fig, filename, dpi=600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_temporal_sdm")
plt.close()
print("Figure A saved.")
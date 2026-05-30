# -*- coding: utf-8 -*-
"""Figure 10: Continuous moderation — industrial structure + institutional complement."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

b1, b2 = 3.8238, -10.1425
se1, se2 = 0.6403, 1.0338
r_corr = -0.95
cov12 = r_corr * se1 * se2

s2 = np.linspace(0.18, 0.70, 300)
me = b1 + b2 * s2
me_se = np.sqrt(se1**2 + s2**2 * se2**2 + 2 * s2 * cov12)
me_lo = me - 1.96 * me_se
me_hi = me + 1.96 * me_se

s2_star = -b1 / b2
s2_e = 0.5028
s2_w = 0.5835
me_e = b1 + b2 * s2_e
me_w = b1 + b2 * s2_w

BLUE  = "#0F4D92"
LIGHT = "#7EB5D6"
FILL  = "#B4C0E4"
RED   = "#B64342"
SOFT  = "#E9A6A1"
GREY  = "#767676"
DARK  = "#272727"

fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
ax.set_facecolor("white")

ax.fill_between(s2, me_lo, me_hi, color=FILL, alpha=0.22, edgecolor="none",
                label="95% 置信区间")
ax.plot(s2, me, color=BLUE, lw=2.2, zorder=4)
ax.axhline(y=0, color=GREY, lw=0.5, ls=":", alpha=0.4)

# Critical point (grey, muted)
ax.axvline(x=s2_star, color=GREY, lw=0.9, ls="--", alpha=0.5)
ax.scatter([s2_star], [0], color=GREY, s=60, zorder=6, edgecolors="white", lw=1.0)
ax.text(s2_star + 0.012, 0.25,
        "临界点 S2* = {:.3f}".format(s2_star),
        fontsize=7.5, color=DARK, fontweight="bold")

# East (red) / West (blue) — consistent with all other figures
ax.axvline(x=s2_e, color=RED, lw=0.7, ls="--", alpha=0.45)
ax.axvline(x=s2_w, color=BLUE, lw=0.7, ls="--", alpha=0.45)
ax.scatter([s2_e], [me_e], color=RED, s=55, zorder=6, edgecolors="white", lw=1.0)
ax.scatter([s2_w], [me_w], color=BLUE, s=55, zorder=6, edgecolors="white", lw=1.0)

ax.text(s2_e - 0.012, me_e + 0.6,
        "东部均值\nS2={:.3f}".format(s2_e),
        fontsize=7, color=RED, fontweight="bold", ha="right")
ax.text(s2_w + 0.012, me_w - 0.6,
        "中西部均值\nS2={:.3f}".format(s2_w),
        fontsize=7, color=BLUE, fontweight="bold", ha="left")

# Region labels
ax.text(0.25, 2.4, "极少数服务型省份\nX→M > 0（京、沪）",
        fontsize=7, color=SOFT, ha="center", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
ax.text(0.66, -1.2, "广大制造业省份\nX→M < 0（新型工业化）",
        fontsize=7, color=LIGHT, ha="center", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

# Institutional annotation
ax.annotate("东部正向驱动\n需叠加制度软环境\n（因果森林：政府效能）",
            xy=(s2_e, me_e),
            xytext=(s2_e + 0.08, me_e + 1.8),
            fontsize=6.8, color=RED, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=SOFT, alpha=0.85, lw=0.5))

# Stats box
stats = ("∂M/∂X = β₁ + β₂ × S2_2011\n"
         "β₁ = {:+.4f}***\n".format(b1) +
         "β₂ = {:+.4f}***\n".format(b2) +
         "R² = 0.9769\n"
         "临界点 S2* = {:.3f}".format(s2_star))
ax.text(0.03, 0.05, stats, transform=ax.transAxes,
        fontsize=7, va="bottom", ha="left", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.9))

ax.set_xlabel("基期二产占比  S2_2011 = 1/(1 + M_2011)", fontsize=9, labelpad=6)
ax.set_ylabel("X 对 M 的边际效应  ∂M/∂X", fontsize=9, labelpad=6)
ax.set_title("遮掩效应的连续调节机制",
             fontsize=11, fontweight="bold", color=DARK, pad=12)

fig.text(0.5, 0.01,
         "注：β₂ < 0 (p<0.001) 证实产业结构是遮掩效应的基础机制。"
         "临界点(S2*≈0.377)偏低 → 仅北京等极少数省份自然越过 → 东部大面积正效应需叠加因果森林识别的\"制度软环境\"驱动。",
         ha="center", fontsize=6.5, color=GREY)

plt.tight_layout(pad=2.0, rect=[0, 0.04, 1, 1])

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_marginal_effect")
plt.close()
print("Figure 10 saved.")

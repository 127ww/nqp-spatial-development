# -*- coding: utf-8 -*-
"""Figure 8: CATE distribution — heterogeneous treatment effects."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from econml.dml import CausalForestDML
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

# ── Load & fit ────────────────────────────────────────────
df = pd.read_csv("data/面板数据_最终测度得分结果.csv", encoding="gbk")
df["ln_Road"] = np.log(df["基础设施_公路里程（公里）"] + 1)
df["ln_Rail"] = np.log(df["基础设施_铁路里程（公里）"] + 1)
df = df.dropna().reset_index(drop=True)

T_col = "新质生产力_组合得分"
Y_col = "高质量发展_组合得分"
ctrl = ["城镇化率（%）", "政府干预度（一般公共预算支出/GDP）", "ln_Road", "ln_Rail"]

prov_w = pd.get_dummies(df["省份"], drop_first=True).astype(float)
year_w = pd.get_dummies(df["年份"], drop_first=True).astype(float)
year_w.columns = year_w.columns.astype(str)
W = pd.concat([prov_w, year_w], axis=1).values

east = ["北京","天津","河北","辽宁","上海","江苏","浙江","福建","山东","广东","海南"]
df["Region"] = df["省份"].apply(lambda x: "东部" if x in east else "中西部")

print("Fitting DML causal forest (~1 min)...")
cf = CausalForestDML(
    model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    n_estimators=1000, min_samples_leaf=10, max_depth=10,
    discrete_treatment=False, random_state=2026, n_jobs=-1, cv=5,
)
cf.fit(Y=df[Y_col].values, T=df[T_col].values,
       X=df[ctrl].values, W=W)

cate = cf.effect(df[ctrl].values)
df["CATE"] = cate
print("Done.")

# ── Build ─────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.5, 4.2), facecolor="white")

# Left: histogram + KDE
ax_a.set_facecolor("white")
ax_a.hist(cate, bins=28, color=BLUE, alpha=0.7, edgecolor="white", lw=0.3,
          density=True)
# KDE
from scipy.stats import gaussian_kde
kde_x = np.linspace(cate.min(), cate.max(), 200)
kde_y = gaussian_kde(cate)(kde_x)
ax_a.plot(kde_x, kde_y, color=RED, lw=1.8)

ax_a.axvline(0, color=GREY, lw=0.6, ls=":")
ax_a.axvline(cate.mean(), color=RED, lw=1.0, ls="--")
ax_a.text(cate.mean() + 0.005, kde_y.max() * 0.85,
          "均值={:.3f}".format(cate.mean()),
          fontsize=7.5, color=RED, fontweight="bold")
ax_a.set_xlabel("条件平均处理效应 (CATE)", fontsize=8)
ax_a.set_ylabel("密度", fontsize=8)
ax_a.set_title("赋能效应分布", fontsize=9, fontweight="bold", color=DARK, loc="left")

# Right: regional comparison
ax_b.set_facecolor("white")
means = [df[df["Region"] == "东部"]["CATE"].mean(),
         df[df["Region"] == "中西部"]["CATE"].mean()]
stds  = [df[df["Region"] == "东部"]["CATE"].std(),
         df[df["Region"] == "中西部"]["CATE"].std()]
colors_bar = [RED, BLUE]
bars = ax_b.bar(["东部", "中西部"], means, color=colors_bar, width=0.45,
                edgecolor="white", lw=0.4)
ax_b.errorbar(["东部", "中西部"], means, yerr=stds, fmt="none",
              ecolor=DARK, capsize=5, lw=1.0)
for bar, val in zip(bars, means):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              "{:.4f}".format(val), ha="center", fontsize=9, fontweight="bold", color=DARK)
ax_b.set_ylabel("平均 CATE", fontsize=8)
ax_b.set_title("区域赋能强度对比", fontsize=9, fontweight="bold", color=DARK, loc="left")

fig.suptitle("新质生产力赋能效应的异质性：后发优势与区域分化",
             fontsize=11, fontweight="bold", color=DARK, y=1.02)

plt.tight_layout(pad=2.0)

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_08_cate")
plt.close()
print("Figure 8 saved.")

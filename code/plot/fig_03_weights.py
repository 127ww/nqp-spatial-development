# -*- coding: utf-8 -*-
"""Figure 3: CRITIC-entropy combined indicator weights."""

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

df = pd.read_excel("data/raw/面板原始数据_合并版v2.xlsx", sheet_name=0)
df["人均技术市场成交额（万元/万人）"] = df["技术市场成交额（万元）"] / df["总人口（万人）"]
df["人均电信业务总量（亿元/万人）"] = df["电信业务总量（亿元）"] / df["总人口（万人）"]
df["单位GDP能耗（万吨标煤/亿元）"] = df["能源消费量（万吨标煤）"] / df["GDP（亿元）"]

# Exact names matching the data file + computed columns
x_cols = [
    "每万人R&D人员（人年/万人）", "人均R&D经费（万元/万人）",
    "每万人专利授权数（件/万人）", "人均电商销售额（亿元/万人）",
    "人均软件产品收入（亿元/万人）", "人均技术市场成交额（万元/万人）",
    "人均电信业务总量（亿元/万人）", "数字普惠金融总指数",
]
# Find medical beds column (encoding may vary)
bed_col = [c for c in df.columns if "床" in c and "每万人" in c][0]

y_pos = [
    "人均地区生产总值（元）", "居民人均可支配收入（元）",
    "对外开放度（进出口/GDP）", "环保重视度（污染治理投资/GDP）",
    bed_col,
]
y_neg = ["单位GDP能耗（万吨标煤/亿元）"]

for col in x_cols + y_pos + y_neg:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for grp in [x_cols, y_pos, y_neg]:
    cols = [c for c in grp if c in df.columns]
    df[cols] = df.groupby("省份")[cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both"))
    df[cols] = df[cols].fillna(df[cols].mean())

def get_weights(data, pos, neg):
    d = data[pos + neg].copy()
    nd = pd.DataFrame(index=d.index, columns=d.columns)
    for c in pos:
        mn, mx = d[c].min(), d[c].max()
        nd[c] = (d[c] - mn) / (mx - mn + 1e-9)
    for c in neg:
        mn, mx = d[c].min(), d[c].max()
        nd[c] = (mx - d[c]) / (mx - mn + 1e-9)
    nsp = nd + 1e-5
    p = nsp / nsp.sum(axis=0)
    n = len(data)
    e = -(1.0 / np.log(n)) * (p * np.log(p)).sum(axis=0)
    w_ent = (1 - e) / (1 - e).sum()
    S = nd.std()
    R = nd.corr()
    C = (1 - R).sum()
    I = S * C
    w_crit = I / I.sum()
    return (w_ent + w_crit) / 2

w_x = get_weights(df, x_cols, [])
w_y = get_weights(df, y_pos, y_neg)

x_labels = ["R&D人员", "R&D经费", "专利授权", "电商销售",
            "软件收入", "技术市场", "电信业务", "数字金融"]
y_labels = ["人均GDP", "可支配收入", "对外开放", "环保投入",
            "医疗床位", "单位能耗\n(负向)"]

BLUE  = "#0F4D92"
RED   = "#B64342"
GREY  = "#767676"
DARK  = "#272727"

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white")
for ax in [ax_l, ax_r]:
    ax.set_facecolor("white")

# Left: X weights
idx_x = np.arange(len(w_x))
ax_l.barh(idx_x, w_x.values, color=BLUE, edgecolor="white", lw=0.3, height=0.6)
ax_l.set_yticks(idx_x)
ax_l.set_yticklabels(x_labels, fontsize=7.5)
ax_l.set_xlabel("权重", fontsize=8)
ax_l.set_title("新质生产力指标权重", fontsize=9, fontweight="bold", color=DARK, loc="left")
for i, v in enumerate(w_x.values):
    ax_l.text(v + 0.005, i, "{:.3f}".format(v), fontsize=7, va="center", color=DARK)

# Right: Y weights (all blue)
idx_y = np.arange(len(w_y))
ax_r.barh(idx_y, w_y.values, color=BLUE, edgecolor="white", lw=0.3, height=0.6)
ax_r.set_yticks(idx_y)
ax_r.set_yticklabels(y_labels, fontsize=7.5)
ax_r.set_xlabel("权重", fontsize=8)
ax_r.set_title("高质量发展指标权重", fontsize=9, fontweight="bold", color=DARK, loc="left")
for i, v in enumerate(w_y.values):
    ax_r.text(v + 0.005, i, "{:.3f}".format(v), fontsize=7, va="center", color=DARK)

fig.suptitle("CRITIC-熵权组合赋权：最终指标权重分布",
             fontsize=11, fontweight="bold", color=DARK, y=1.02)

plt.tight_layout(pad=2.0)

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "output/fig_03_weights")
plt.close()
print("Figure 3 saved.")

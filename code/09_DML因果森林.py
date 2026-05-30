import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from econml.dml import CausalForestDML

df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='gbk', engine='python')
df['ln_Road'] = np.log(df['基础设施_公路里程（公里）'] + 1)
df['ln_Rail'] = np.log(df['基础设施_铁路里程（公里）'] + 1)
df = df.dropna().reset_index(drop=True)

T_col = '新质生产力_组合得分'
Y_col = '高质量发展_组合得分'
cols_control = ['城镇化率（%）', '政府干预度（一般公共预算支出/GDP）', 'ln_Road', 'ln_Rail']

prov_dummies = pd.get_dummies(df['省份'], drop_first=True).astype(float)
year_dummies = pd.get_dummies(df['年份'], drop_first=True).astype(float)
year_dummies.columns = [str(c) for c in year_dummies.columns]

T = df[T_col].values
Y = df[Y_col].values
X = df[cols_control].values
W = pd.concat([prov_dummies, year_dummies], axis=1).values

print(f"样本量: {len(df)}, 特征数: {X.shape[1]}, 混杂维度: {W.shape[1]}")
print(f"T: mean={T.mean():.4f}, std={T.std():.4f}")
print(f"Y: mean={Y.mean():.4f}, std={Y.std():.4f}")

cf = CausalForestDML(
    model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    n_estimators=1000, min_samples_leaf=10, max_depth=10,
    discrete_treatment=False, random_state=2026, n_jobs=-1, cv=5
)

print("\nFitting DML causal forest...")
cf.fit(Y=Y, T=T, X=X, W=W)
print("Done.")

cate = cf.effect(X)
cate_lower, cate_upper = cf.effect_interval(X, alpha=0.05)
df['CATE'] = cate
df['CATE_lower'] = cate_lower
df['CATE_upper'] = cate_upper

east_list = ['北京', '天津', '河北', '辽宁', '上海', '江苏',
             '浙江', '福建', '山东', '广东', '海南']
df['Region'] = df['省份'].apply(lambda x: '东部' if x in east_list else '中西部')

imp_global = pd.DataFrame({
    'Feature': cols_control,
    'Importance': cf.feature_importances_
}).sort_values('Importance', ascending=False)

df_east = df[df['Region'] == '东部']
df_west = df[df['Region'] == '中西部']

cf_east = CausalForestDML(
    model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    n_estimators=1000, min_samples_leaf=10, max_depth=10,
    discrete_treatment=False, random_state=2026, n_jobs=-1, cv=5
)
cf_east.fit(Y=df_east[Y_col].values, T=df_east[T_col].values,
            X=df_east[cols_control].values,
            W=pd.concat([pd.get_dummies(df_east['省份'], drop_first=True).astype(float),
                         pd.get_dummies(df_east['年份'], drop_first=True).astype(float)], axis=1).values)

cf_west = CausalForestDML(
    model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2026),
    n_estimators=1000, min_samples_leaf=10, max_depth=10,
    discrete_treatment=False, random_state=2026, n_jobs=-1, cv=5
)
cf_west.fit(Y=df_west[Y_col].values, T=df_west[T_col].values,
            X=df_west[cols_control].values,
            W=pd.concat([pd.get_dummies(df_west['省份'], drop_first=True).astype(float),
                         pd.get_dummies(df_west['年份'], drop_first=True).astype(float)], axis=1).values)

imp_east = pd.DataFrame({
    'Feature': cols_control,
    '东部重要度': cf_east.feature_importances_
}).sort_values('东部重要度', ascending=False)

imp_west = pd.DataFrame({
    'Feature': cols_control,
    '中西部重要度': cf_west.feature_importances_
}).sort_values('中西部重要度', ascending=False)

PRIMARY = '#0F4D92'
EAST    = '#B64342'
MIDWEST = '#3775BA'
REF_RED = '#B64342'
BG      = '#FFFFFF'
EDGE    = '#272727'

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Arial", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)

ax = axes[0]
ax.set_facecolor(BG)
sns.histplot(df['CATE'], kde=True, bins=30, color=PRIMARY,
             edgecolor='white', linewidth=0.5, alpha=0.85, ax=ax)
ax.axvline(0, color=REF_RED, linestyle='--', linewidth=1.5, alpha=0.85)
ylim = ax.get_ylim()
ax.axvline(df['CATE'].mean(), color=EAST, linestyle='-', linewidth=1.5)
ax.text(df['CATE'].mean() + 0.003, ylim[1] * 0.85,
        f'均值={df["CATE"].mean():.4f}', color=EAST, fontsize=9, fontweight='bold')
ax.set_xlabel('条件平均处理效应 (CATE)', fontsize=11)
ax.set_ylabel('频数', fontsize=11)
ax.set_title('赋能效应分布', fontsize=13, fontweight='bold', loc='left')

ax = axes[1]
ax.set_facecolor(BG)
regions = ['东部', '中西部']
cate_means = [df[df['Region'] == r]['CATE'].mean() for r in regions]
cate_std = [df[df['Region'] == r]['CATE'].std() for r in regions]
bars = ax.bar(regions, cate_means, color=[EAST, MIDWEST], edgecolor=EDGE,
              linewidth=1.0, width=0.5)
ax.errorbar(regions, cate_means, yerr=cate_std, fmt='none',
            ecolor=EDGE, capsize=8, linewidth=1.2)
for bar, val, std in zip(bars, cate_means, cate_std):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.008,
            f'{val:.4f}', ha='center', fontsize=13, fontweight='bold', color=EDGE)
ax.set_ylabel('平均 CATE', fontsize=11)
ax.set_title('分区域赋能效应对比', fontsize=13, fontweight='bold', loc='left')
ax.set_ylim(0, max(cate_means) * 1.35)

plt.tight_layout(pad=2)

def save_pub_py(fig, name, dpi=600):
    fig.savefig(f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

save_pub_py(fig, "cate_distribution_dml")
plt.close()
print("Figure saved.")

print("\n" + "=" * 60)
print("CATE 统计描述")
print("=" * 60)
print(df['CATE'].describe())
print(f"\n负向样本: {(df['CATE'] < 0).mean() * 100:.2f}%")
print(f"显著正向: {(df['CATE_lower'] > 0).mean() * 100:.2f}%")

print("\n全局特征重要度:")
for _, row in imp_global.iterrows():
    print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

print("\n区域特征重要度对比:")
merged_imp = imp_east.merge(imp_west, on='Feature')
print(f"{'特征':<30} {'东部':>10} {'中西部':>10}")
print("-" * 52)
for _, row in merged_imp.iterrows():
    print(f"{row['Feature']:<30} {row['东部重要度']:>10.4f} {row['中西部重要度']:>10.4f}")

print("\n区域 CATE:")
print(df.groupby('Region')['CATE'].describe())

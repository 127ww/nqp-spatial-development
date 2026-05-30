import pandas as pd
import numpy as np
from scipy import stats, optimize

df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='gbk', engine='python')
w_adj = pd.read_csv('data/W_adj_norm.csv', header=None).values
N_cross = w_adj.shape[0]

df['ln_Road'] = np.log(df['基础设施_公路里程（公里）'] + 1)
df['ln_Rail'] = np.log(df['基础设施_铁路里程（公里）'] + 1)

Y_col = '高质量发展_组合得分'
X_col = '新质生产力_组合得分'
controls = ['城镇化率（%）', '政府干预度（一般公共预算支出/GDP）', 'ln_Road', 'ln_Rail']

years = sorted(df['年份'].unique())
provinces = sorted(df['省份'].unique())
N = len(df)

def spat_lag_col(df, col, W):
    vals = np.zeros(len(df))
    for t in years:
        idx = df['年份'] == t
        vals[idx] = W.dot(df.loc[idx, col].values)
    return vals

df['WY'] = spat_lag_col(df, Y_col, w_adj)
df['WX'] = spat_lag_col(df, X_col, w_adj)

Z_cols = [X_col, 'WX'] + controls
prov_dummies = pd.get_dummies(df['省份'], drop_first=True).astype(float)
year_dummies = pd.get_dummies(df['年份'], drop_first=True).astype(float)

Z_base = pd.concat([df[Z_cols], prov_dummies, year_dummies], axis=1)
Z_names = list(Z_base.columns)
Z_mat = Z_base.values
y_vec = df[Y_col].values
WY_vec = df['WY'].values

K = Z_mat.shape[1]
print(f"样本量 N = {N}, 参数数 K = {K}")

eigvals = np.linalg.eigvals(w_adj)
eigvals_real = eigvals.real
print(f"W 特征值范围: [{eigvals_real.min():.4f}, {eigvals_real.max():.4f}]")

def log_det_A(rho):
    vals = 1 - rho * eigvals_real
    if np.any(vals <= 0):
        return -np.inf
    return np.sum(np.log(np.abs(vals)))

def concentrated_loglik(rho, y, Wy, Z):
    det_term = log_det_A(rho)
    if np.isneginf(det_term):
        return -1e15
    Ay = y - rho * Wy
    try:
        ZtZ_inv = np.linalg.solve(Z.T @ Z, np.eye(K))
        delta_hat = ZtZ_inv @ (Z.T @ Ay)
        residuals = Ay - Z @ delta_hat
        sigma2 = np.sum(residuals ** 2) / len(y)
    except np.linalg.LinAlgError:
        return -1e15
    if sigma2 <= 0:
        return -1e15
    loglik = -0.5 * N * np.log(2 * np.pi) - 0.5 * N * np.log(sigma2) + det_term - 0.5 * N
    return loglik

lambda_min, lambda_max = eigvals_real.min(), eigvals_real.max()
rho_lower = max(1.0 / lambda_min if lambda_min < 0 else -0.999, -0.999)
rho_upper = min(1.0 / lambda_max if lambda_max > 0 else 0.999, 0.999)
print(f"rho 搜索区间: [{rho_lower:.4f}, {rho_upper:.4f}]")

grid = np.linspace(rho_lower + 0.001, rho_upper - 0.001, 50)
best_rho = grid[0]
best_ll = -1e15
for r in grid:
    ll = concentrated_loglik(r, y_vec, WY_vec, Z_mat)
    if ll > best_ll:
        best_ll = ll
        best_rho = r
print(f"网格搜索初始 rho = {best_rho:.4f}, ln L = {best_ll:.4f}")

result = optimize.minimize(
    lambda r: -concentrated_loglik(r[0], y_vec, WY_vec, Z_mat),
    x0=[best_rho],
    bounds=[(rho_lower + 1e-6, rho_upper - 1e-6)],
    method='L-BFGS-B'
)

rho_mle = result.x[0]
ll_mle = -result.fun
print(f"MLE 最优 rho = {rho_mle:.4f}, ln L = {ll_mle:.4f}")

Ay = y_vec - rho_mle * WY_vec
ZtZ_inv = np.linalg.solve(Z_mat.T @ Z_mat, np.eye(K))
delta_hat = ZtZ_inv @ (Z_mat.T @ Ay)
residuals = Ay - Z_mat @ delta_hat
sigma2_mle = np.sum(residuals ** 2) / N

delta_se = np.sqrt(sigma2_mle * np.diag(ZtZ_inv))

eps = 1e-5
ll_plus = concentrated_loglik(rho_mle + eps, y_vec, WY_vec, Z_mat)
ll_minus = concentrated_loglik(rho_mle - eps, y_vec, WY_vec, Z_mat)
ll_0 = concentrated_loglik(rho_mle, y_vec, WY_vec, Z_mat)
d2_rho = (ll_plus - 2 * ll_0 + ll_minus) / (eps ** 2)
rho_se = np.sqrt(-1.0 / d2_rho) if d2_rho < 0 else np.nan

beta_x = delta_hat[list(Z_names).index(X_col)]
theta_wx = delta_hat[list(Z_names).index('WX')]

I_N = np.eye(N_cross)
try:
    inv_A = np.linalg.inv(I_N - rho_mle * w_adj)
except np.linalg.LinAlgError:
    inv_A = np.linalg.pinv(I_N - rho_mle * w_adj)

effect_mat = inv_A @ (beta_x * I_N + theta_wx * w_adj)
direct_effect = np.trace(effect_mat) / N_cross
total_effect = np.sum(effect_mat) / N_cross
indirect_effect = total_effect - direct_effect

np.random.seed(20260503)
cov_par = np.zeros((3, 3))
cov_block = sigma2_mle * ZtZ_inv[:2, :2]
cov_par[:2, :2] = cov_block
cov_par[2, 2] = rho_se ** 2 if not np.isnan(rho_se) else 0

n_sim = 5000
indirect_sims = np.zeros(n_sim)
direct_sims = np.zeros(n_sim)
mean_par = np.array([beta_x, theta_wx, rho_mle])
valid = 0

for i in range(n_sim):
    try:
        s = np.random.multivariate_normal(mean_par, cov_par)
        inv_s = np.linalg.inv(I_N - s[2] * w_adj)
        eff_s = inv_s @ (s[0] * I_N + s[1] * w_adj)
        direct_sims[i] = np.trace(eff_s) / N_cross
        indirect_sims[i] = np.sum(eff_s) / N_cross - direct_sims[i]
        valid += 1
    except np.linalg.LinAlgError:
        pass

if valid > 0:
    direct_sims = direct_sims[direct_sims != 0] if np.any(direct_sims != 0) else direct_sims[:valid]
    indirect_sims = indirect_sims[:valid]
    direct_se = np.std(direct_sims) if len(direct_sims) > 1 else 0
    indirect_se = np.std(indirect_sims) if len(indirect_sims) > 1 else 0
    indirect_z = indirect_effect / indirect_se if indirect_se > 0 else 0
    indirect_pval = 2 * (1 - stats.norm.cdf(abs(indirect_z)))
else:
    direct_se = indirect_se = indirect_pval = np.nan

def sstars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.1: return '*'
    return ''

print("\n" + "=" * 70)
print("表4：SDM 基准回归与效应分解")
print("=" * 70)
print(f"\n  估计方法: 集中极大似然估计 (Concentrated MLE)")
print(f"  对数似然值: {ll_mle:.4f}")
print(f"  sigma2: {sigma2_mle:.6f}")
print(f"\n  {'变量':<28} {'系数':>10} {'标准误':>10} {'z值':>10}")
print(f"  {'-'*62}")

z_rho = rho_mle / rho_se if rho_se and rho_se > 0 else 0
p_rho = 2 * (1 - stats.norm.cdf(abs(z_rho)))
print(f"  {'因变量空间滞后 (WY)':<28} {rho_mle:>10.4f} {rho_se:>10.4f} {z_rho:>10.2f}{sstars(p_rho)}")

for name in [X_col, 'WX'] + controls:
    idx = list(Z_names).index(name)
    coef = delta_hat[idx]
    se = delta_se[idx]
    z = coef / se if se > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    label = '自变量空间滞后 (WX)' if name == 'WX' else name
    print(f"  {label:<28} {coef:>10.4f} {se:>10.4f} {z:>10.2f}{sstars(p)}")

print(f"\n  {'LeSage & Pace 效应分解':-^55}")
print(f"  {'直接效应 (Direct)':<28} {direct_effect:>10.4f}")
print(f"  {'间接效应 (Indirect)':<28} {indirect_effect:>10.4f}")
print(f"  {'总效应 (Total)':<28} {total_effect:>10.4f}")

if indirect_se and indirect_se > 0:
    print(f"\n  间接效应 Monte Carlo SE: {indirect_se:.4f}")
    print(f"  间接效应 p值:            {indirect_pval:.4f}")
    ci_lo = indirect_effect - 1.96 * indirect_se
    ci_hi = indirect_effect + 1.96 * indirect_se
    print(f"  间接效应 95% CI:         [{ci_lo:.4f}, {ci_hi:.4f}]")

import statsmodels.api as sm
X_ols = pd.concat([df[['WY', X_col, 'WX'] + controls], prov_dummies, year_dummies], axis=1)
ols_model = sm.OLS(y_vec, sm.add_constant(X_ols)).fit()

print(f"\n  {'OLS vs MLE 对比':-^55}")
print(f"  {'参数':<28} {'OLS (LSDV)':>12} {'MLE':>12}")
print(f"  {'-'*52}")
print(f"  {'因变量空间滞后 (WY)':<28} {ols_model.params['WY']:>12.4f} {rho_mle:>12.4f}")
print(f"  {'新质生产力 (X)':<28} {ols_model.params[X_col]:>12.4f} {beta_x:>12.4f}")
print(f"  {'自变量空间滞后 (WX)':<28} {ols_model.params['WX']:>12.4f} {theta_wx:>12.4f}")
print(f"\n  注：OLS 估计 rho 通常下偏，MLE 通过雅可比项校正。")


def run_sdm_subperiod(df_full, year_range, w_adj):
    sub = df_full[df_full['年份'].between(year_range[0], year_range[1])].copy()
    N_sub = len(sub)
    N_cross = w_adj.shape[0]

    def spat_lag_sub(data, col, W):
        vals = np.zeros(len(data))
        for t in sorted(data['年份'].unique()):
            idx = data['年份'] == t
            vals[idx] = W.dot(data.loc[idx, col].values)
        return vals

    sub['WY'] = spat_lag_sub(sub, Y_col, w_adj)
    sub['WX'] = spat_lag_sub(sub, X_col, w_adj)

    Z_sub = pd.concat([
        sub[[X_col, 'WX'] + controls],
        pd.get_dummies(sub['省份'], drop_first=True).astype(float),
        pd.get_dummies(sub['年份'], drop_first=True).astype(float)
    ], axis=1)
    Z_mat_sub = Z_sub.values
    Z_names_sub = list(Z_sub.columns)
    y_sub = sub[Y_col].values
    WY_sub = sub['WY'].values

    eigvals = np.linalg.eigvals(w_adj)
    eigvals_real = eigvals.real

    def log_det_A_sub(rho):
        vals = 1 - rho * eigvals_real
        if np.any(vals <= 0):
            return -np.inf
        return np.sum(np.log(np.abs(vals)))

    def cll_sub(rho, y, Wy, Z):
        det = log_det_A_sub(rho)
        if np.isneginf(det):
            return -1e15
        Ay = y - rho * Wy
        try:
            ZtZ_inv = np.linalg.solve(Z.T @ Z, np.eye(Z.shape[1]))
            delta = ZtZ_inv @ (Z.T @ Ay)
            resid = Ay - Z @ delta
            s2 = np.sum(resid ** 2) / len(y)
        except np.linalg.LinAlgError:
            return -1e15
        if s2 <= 0:
            return -1e15
        return -0.5 * len(y) * np.log(2 * np.pi) - 0.5 * len(y) * np.log(s2) + det - 0.5 * len(y)

    lambda_min, lambda_max = eigvals_real.min(), eigvals_real.max()
    rlo = max(1.0 / lambda_min if lambda_min < 0 else -0.999, -0.999)
    rhi = min(1.0 / lambda_max if lambda_max > 0 else 0.999, 0.999)

    grid = np.linspace(rlo + 0.001, rhi - 0.001, 50)
    brho = grid[0]; bll = -1e15
    for r in grid:
        ll = cll_sub(r, y_sub, WY_sub, Z_mat_sub)
        if ll > bll:
            bll = ll; brho = r

    res = optimize.minimize(
        lambda r: -cll_sub(r[0], y_sub, WY_sub, Z_mat_sub),
        x0=[brho], bounds=[(rlo + 1e-6, rhi - 1e-6)], method='L-BFGS-B'
    )
    rho_mle = res.x[0]; ll_mle = -res.fun

    Ay = y_sub - rho_mle * WY_sub
    ZtZ_inv = np.linalg.solve(Z_mat_sub.T @ Z_mat_sub, np.eye(Z_mat_sub.shape[1]))
    delta_hat = ZtZ_inv @ (Z_mat_sub.T @ Ay)
    resid = Ay - Z_mat_sub @ delta_hat
    s2_mle = np.sum(resid ** 2) / N_sub

    beta_x = delta_hat[list(Z_names_sub).index(X_col)]
    theta_wx = delta_hat[list(Z_names_sub).index('WX')]

    eps = 1e-5
    ll_p = cll_sub(rho_mle + eps, y_sub, WY_sub, Z_mat_sub)
    ll_m = cll_sub(rho_mle - eps, y_sub, WY_sub, Z_mat_sub)
    ll_0 = cll_sub(rho_mle, y_sub, WY_sub, Z_mat_sub)
    d2 = (ll_p - 2 * ll_0 + ll_m) / (eps ** 2)
    rho_se = np.sqrt(-1.0 / d2) if d2 < 0 else np.nan

    I_N = np.eye(N_cross)
    try:
        inv_A = np.linalg.inv(I_N - rho_mle * w_adj)
    except np.linalg.LinAlgError:
        inv_A = np.linalg.pinv(I_N - rho_mle * w_adj)
    eff_mat = inv_A @ (beta_x * I_N + theta_wx * w_adj)
    direct_eff = np.trace(eff_mat) / N_cross
    total_eff = np.sum(eff_mat) / N_cross
    indirect_eff = total_eff - direct_eff

    cov_par = np.zeros((3, 3))
    cov_block = s2_mle * ZtZ_inv[:2, :2]
    cov_par[:2, :2] = cov_block
    cov_par[2, 2] = rho_se ** 2 if (rho_se and not np.isnan(rho_se)) else 0

    np.random.seed(20260503)
    n_sim = 5000
    indir_sims = np.zeros(n_sim)
    direct_sims = np.zeros(n_sim)
    mean_par = np.array([beta_x, theta_wx, rho_mle])
    valid = 0
    for i in range(n_sim):
        try:
            s = np.random.multivariate_normal(mean_par, cov_par)
            inv_s = np.linalg.inv(I_N - s[2] * w_adj)
            eff_s = inv_s @ (s[0] * I_N + s[1] * w_adj)
            direct_sims[i] = np.trace(eff_s) / N_cross
            indir_sims[i] = np.sum(eff_s) / N_cross - direct_sims[i]
            valid += 1
        except np.linalg.LinAlgError:
            pass

    if valid > 0:
        indir_sims = indir_sims[:valid]
        indir_se = np.std(indir_sims)
        indir_z = indirect_eff / indir_se if indir_se > 0 else 0
        indir_p = 2 * (1 - stats.norm.cdf(abs(indir_z)))
    else:
        indir_se = indir_p = np.nan

    x_se = np.sqrt(s2_mle * np.diag(ZtZ_inv))[list(Z_names_sub).index(X_col)]
    x_z = beta_x / x_se if x_se > 0 else 0
    x_p = 2 * (1 - stats.norm.cdf(abs(x_z)))

    return {
        'N': N_sub, 'rho': rho_mle, 'rho_se': rho_se,
        'beta_x': beta_x, 'x_se': x_se, 'x_p': x_p,
        'theta_wx': theta_wx,
        'direct': direct_eff, 'indirect': indirect_eff,
        'indirect_se': indir_se, 'indirect_p': indir_p,
        'total': total_eff, 'll': ll_mle
    }


print("\n" + "=" * 85)
print("表5：分时段 SDM 效应分解")
print("=" * 85)

res_early = run_sdm_subperiod(df, (2011, 2017), w_adj)
res_late  = run_sdm_subperiod(df, (2018, 2023), w_adj)

for r, name in [(res_early, '2011-2017'), (res_late, '2018-2023')]:
    print(f"\n  {name}: N={r['N']}, 对数似然={r['ll']:.2f}")
    print(f"  {'直接效应':<20} {r['direct']:>10.4f}")
    print(f"  {'间接效应':<20} {r['indirect']:>10.4f}  (SE={r['indirect_se']:.4f}, p={r['indirect_p']:.4f})")
    print(f"  {'总效应':<20} {r['total']:>10.4f}")
    print(f"  {'WX (theta)':<20} {r['theta_wx']:>10.4f}")
    print(f"  {'WY (rho)':<20} {r['rho']:>10.4f}  (SE={r['rho_se']:.4f})")

delta_indirect = res_late['indirect'] - res_early['indirect']
print(f"\n  间接效应变化: {delta_indirect:+.4f}")

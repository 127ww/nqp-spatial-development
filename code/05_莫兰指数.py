import pandas as pd
import numpy as np
from scipy import stats

# ==========================================
# 1. 读取数据与空间权重矩阵
# ==========================================
try:
    df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='gbk')
except:
    df = pd.read_csv('data/面板数据_最终测度得分结果.csv', encoding='utf-8-sig')

# 读取 31x31 的空间权重矩阵（已行标准化）
w_df = pd.read_csv('data/W_adj_norm.csv', header=None)

# 统一省份标准顺序
standard_provinces = [
    '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
    '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
    '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
]
w_df.index = standard_provinces
w_df.columns = standard_provinces


# 清洗面板数据里的省份名称
def clean_name(name):
    return str(name).replace('省', '').replace('市', '').replace('自治区', '').replace('维吾尔', '').replace('回族',
                                                                                                             '').replace(
        '壮族', '')


df['省份'] = df['省份'].apply(clean_name)

# ==========================================
# 2. 构建行标准化的空间权重矩阵 W
# ==========================================
W_norm = w_df.values.astype(float)
# data/W_adj_norm.csv 已是行标准化矩阵，直接使用


# ==========================================
# 3. 核心计算函数
# ==========================================
def calculate_moran_stats(data, weight_matrix, column_name):
    """
    计算莫兰指数、Z值(统计量)和P值
    """
    x = data[column_name].values

    # 1. 标准化 (Z-score)
    z = (x - np.mean(x)) / np.std(x)
    # 2. 计算空间滞后项
    wz = np.dot(weight_matrix, z)

    # 3. 回归计算
    slope, intercept, r_value, p_value, std_err = stats.linregress(z, wz)

    # 在标准化的OLS回归框架下，检验莫兰指数显著性的 Z值/T值 即为：斜率 / 标准误
    z_value = slope / std_err

    return slope, z_value, p_value


# ==========================================
# 4. 循环计算每年、双变量结果
# ==========================================
results = []
years = sorted(df['年份'].unique())

for year in years:
    # 提取当年数据，并严格按照 31 省顺序对齐
    df_year = df[df['年份'] == year].set_index('省份').reindex(standard_provinces)

    # 计算：新质生产力
    mi_xz, z_xz, p_xz = calculate_moran_stats(df_year, W_norm, '新质生产力_组合得分')

    # 计算：高质量发展
    mi_gz, z_gz, p_gz = calculate_moran_stats(df_year, W_norm, '高质量发展_组合得分')

    # 保存结果
    results.append({
        '年份': year,
        '新质_莫兰指数': round(mi_xz, 4),
        '新质_Z值': round(z_xz, 4),
        '新质_P值': round(p_xz, 6),
        '高质_莫兰指数': round(mi_gz, 4),
        '高质_Z值': round(z_gz, 4),
        '高质_P值': round(p_gz, 6)
    })

# ==========================================
# 5. 输出展示与保存
# ==========================================
res_df = pd.DataFrame(results)

print("============ 历年全局莫兰指数计算结果 ============")
print(res_df.to_string(index=False))
print("==================================================")

# 将结果保存为 Excel，方便你放入论文中
res_df.to_excel('历年_莫兰指数_双变量结果.xlsx', index=False)
print("\n✅ 计算完成！结果已保存为 '历年_莫兰指数_双变量结果.xlsx'")
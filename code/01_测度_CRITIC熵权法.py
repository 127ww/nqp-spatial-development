import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('data/raw/面板原始数据_合并版v2.xlsx', sheet_name=0)

# 1. 补全均量指标
df['人均技术市场成交额（万元/万人）'] = df['技术市场成交额（万元）'] / df['总人口（万人）']
df['人均电信业务总量（亿元/万人）'] = df['电信业务总量（亿元）'] / df['总人口（万人）']
df['单位GDP能耗（万吨标煤/亿元）'] = df['能源消费量（万吨标煤）'] / df['GDP（亿元）']

# 2. 定义指标阵营
x_cols_pos = [
    '每万人R&D人员（人年/万人）', '人均R&D经费（万元/万人）',
    '每万人专利授权数（件/万人）', '人均电商销售额（亿元/万人）',
    '人均软件产品收入（亿元/万人）', '人均技术市场成交额（万元/万人）',
    '人均电信业务总量（亿元/万人）', '数字普惠金融总指数'
]

y_cols_pos = [
    '人均地区生产总值（元）', '居民人均可支配收入（元）',
    '对外开放度（进出口/GDP）', '环保重视度（污染治理投资/GDP）',
    '医疗床位数（张/万人）'
]
y_cols_neg = ['单位GDP能耗（万吨标煤/亿元）']

# 打印一下当前的列名，检查匹配情况
print("Data columns:", df.columns.tolist())

# 检查列名是否存在，不存在的话可能是命名有微调
actual_x_cols = [col for col in x_cols_pos if col in df.columns]
actual_y_pos = [col for col in y_cols_pos if col in df.columns]
actual_y_neg = [col for col in y_cols_neg if col in df.columns]

# 3. 缺失值处理
cols_to_process = actual_x_cols + actual_y_pos + actual_y_neg
for col in cols_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[cols_to_process] = df.groupby('省份')[cols_to_process].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)
df[cols_to_process] = df[cols_to_process].fillna(df[cols_to_process].mean())


# 4. 组合赋权法
def combined_weighting(df_data, pos_cols, neg_cols=[]):
    data = df_data[pos_cols + neg_cols].copy()
    norm_data = pd.DataFrame(index=data.index, columns=data.columns)

    # 标准化
    for col in pos_cols:
        min_v, max_v = data[col].min(), data[col].max()
        norm_data[col] = (data[col] - min_v) / (max_v - min_v + 1e-9)

    for col in neg_cols:
        min_v, max_v = data[col].min(), data[col].max()
        norm_data[col] = (max_v - data[col]) / (max_v - min_v + 1e-9)

    # 熵权法
    norm_data_shifted = norm_data + 1e-5
    p = norm_data_shifted / norm_data_shifted.sum(axis=0)
    n = len(data)
    e = - (1.0 / np.log(n)) * (p * np.log(p)).sum(axis=0)
    w_entropy = (1 - e) / (1 - e).sum()

    # CRITIC法
    S = norm_data.std()
    R = norm_data.corr()
    C = (1 - R).sum()
    I = S * C
    w_critic = I / I.sum()

    # 组合权重
    w_combined = (w_entropy + w_critic) / 2
    score = (norm_data * w_combined).sum(axis=1)

    weights_df = pd.DataFrame({
        '指标': pos_cols + neg_cols,
        '熵权法权重': w_entropy.values,
        'CRITIC权重': w_critic.values,
        '最终组合权重': w_combined.values
    })
    return score, weights_df


# 计算得分
df['新质生产力_组合得分'], w_x_df = combined_weighting(df, actual_x_cols, [])
df['高质量发展_组合得分'], w_y_df = combined_weighting(df, actual_y_pos, actual_y_neg)

print("========= 新质生产力 (X) 权重 =========")
print(w_x_df)
print("\n========= 高质量发展 (Y) 权重 =========")
print(w_y_df)

# 保存最终文件
final_output_cols = [
                        '省份', '年份', '新质生产力_组合得分', '高质量发展_组合得分',
                        '产业高级化率（%）', '城镇化率（%）', '政府干预度（一般公共预算支出/GDP）',
                        '基础设施_公路里程（公里）', '基础设施_铁路里程（公里）'
                    ] + actual_x_cols + actual_y_pos + actual_y_neg

df_final = df[final_output_cols]
output_file = 'data/面板数据_最终测度得分结果.csv'
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n成功生成测度结果文件：{output_file}")
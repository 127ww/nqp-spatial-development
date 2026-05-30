# 极化、涓滴与遮掩：新质生产力赋能高质量发展的时空效应识别

基于2011-2023年中国31省份面板数据，综合运用**空间杜宾模型(SDM)**、**空间联立方程(SSEM)** 与**双重机器学习因果森林(DML + Causal Forest)**，系统剖析新质生产力赋能高质量发展的时空博弈与传导机制。

## 核心发现

- **涓滴退化**：空间溢出从前期的显著正向(+0.357, p<0.05)逆转为后期的显著负向(-0.142, p<0.05)
- **遮掩效应**：基期产业结构是遮掩的基础条件(交互项β=-10.14, p<0.001)，东部正向驱动需叠加制度软环境
- **后发优势**：中西部边际赋能强度(0.388)显著高于东部(0.305)

## 结构

```
├── README.md
├── 论文.docx
├── code/
│   ├── 01_测度_CRITIC熵权法.py       # 综合测度
│   ├── 02_三维核密度图.py            # 动态演进
│   ├── 03_箱线图与条形图.py          # 区域差异
│   ├── 04_三大地带折线图.py          # 演进趋势
│   ├── 05_莫兰指数.py               # 全局空间自相关
│   ├── 06_莫兰散点图.py              # 局部空间自相关
│   ├── 07_SDM_MLE基准回归.py         # SDM + 分时段
│   ├── 08_SSEM_2SLS联立方程.py       # SSEM + 连续调节
│   ├── 09_DML因果森林.py             # 因果森林异质性
│   ├── 10_稳健性检验.py              # 2SLS + GMM
│   └── plot/                         # 图表生成脚本
├── data/
│   ├── 面板数据_最终测度得分结果.csv
│   ├── W_adj_norm.csv / W_eco_norm.csv / W_geo_norm.csv
│   └── raw/                          # 原始数据
└── output/                           # 输出图表 (PNG+SVG+PDF)
```

## 复现

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels econml scikit-learn

cd code
python 01_测度_CRITIC熵权法.py
python 05_莫兰指数.py
python 07_SDM_MLE基准回归.py
python 08_SSEM_2SLS联立方程.py
python 09_DML因果森林.py
```

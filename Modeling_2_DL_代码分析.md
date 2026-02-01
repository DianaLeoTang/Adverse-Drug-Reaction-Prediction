# Modeling 2-DL 代码分析

本文档对应 `Modeling 2-DL.ipynb` / `modeling_2_dl.py`，按执行顺序说明各段代码的作用。数据与划分方式与 Modeling 1-RF 一致，仅模型改为深度学习（Keras 多标签分类）。

---

## 1. 数据加载与按分子簇划分（与 RF 一致）

| 代码/步骤 | 作用 |
|-----------|------|
| `pd.read_csv("data.csv")` | 读取合并后的主表（化学 + 分子 + 生物 + 人口统计 + 30 种 ADR 标签）。 |
| `data.loc[:, "molecular_weight":"covalent_unit_count"]` | 提取分子描述符子表，用于聚类。 |
| `KMeans(n_clusters=25, random_state=42).fit(molecular_df)` | 在分子特征上做 K-means，得到 25 个分子簇。 |
| `tmp.sample(frac=1, random_state=30).cumsum()` | 随机打乱簇顺序后累加占比，用于 80/20 划分。 |
| `train_clusters` / `test_clusters` | 累计占比 < 80% 的簇为训练簇，≥ 80% 为测试簇。 |
| `train_mol` / `test_mol` | 根据簇标签得到训练/测试对应的分子行索引。 |
| `data.loc[:, "molecular_weight":"covalent_unit_count"].isin(train_mol).all(axis=1)` | 根据分子行是否属于 train_mol/test_mol，给每条样本打 `type='train'` 或 `'test'`。 |

**目的**：在分子类别层面做 80/20 划分，防止同一分子类别同时出现在训练集和测试集中（防泄漏）。

---

## 2. 分子特征标准化与冗余列删除

| 代码/步骤 | 作用 |
|-----------|------|
| `Normalizer()` + `fit_transform(...)` | 对分子描述符做行向量的 L2 归一化。 |
| `data.drop(["sex_F", "age_group_1"], axis=1, inplace=True)` | 删除冗余人口统计列。 |
| `train = data[data["type"] == "train"]` | 按 `type` 拆出训练集。 |
| `test = data[data["type"] == "test"]` | 按 `type` 拆出测试集。 |

---

## 3. 深度学习多标签实验：不同特征组合

### 3.1 网络结构与训练流程

| 代码/步骤 | 作用 |
|-----------|------|
| `Sequential()` | Keras 顺序模型。 |
| `Dense(512, activation='relu', input_shape=(x_train.shape[1],))` | 输入层 + 第一隐藏层，512 单元，ReLU。 |
| `Dropout(0.3~0.7)` | Dropout 防过拟合。 |
| `Dense(256, activation='relu')` | 第二隐藏层，256 单元，ReLU。 |
| `Dense(30, activation='sigmoid')` | 输出层，30 个节点对应 30 种 ADR，Sigmoid 做多标签二分类。 |
| `model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])` | 损失为二值交叉熵，指标为 AUC（与文章设计一致）。 |
| `model.fit(x_train, y_train, epochs=..., batch_size=..., validation_split=0.2)` | 训练，部分小节使用 20% 验证集。 |
| `model.predict(x_test)` | 在测试集上得到 30 维预测概率。 |
| `roc_auc_score(y_test, predictions, average=None)` | 每个 ADR 的 AUC。 |
| `roc_auc_score(..., average='macro')` | 宏平均 AUC。 |
| `average_precision_score(..., average=None/'macro')` | 每类及宏平均 MAP。 |

**与文章对应**：4 层结构（输入 + 2 隐藏层 + 输出）、30 个输出节点、Sigmoid、交叉熵、评估 AUC 与 MAP。

### 3.2 各小节特征选取

各小节仅 **特征 X 的列选取** 不同，标签统一为 `cardiac failure` ~ `vomiting`（30 个 ADR）。

| 小节 | 特征选取 | 说明 |
|------|----------|------|
| **ALL** | `train.loc[:, 'sex_M':"880"]` | 全部特征；部分 cell 含 Grid Search（optimizer、dropout_rate、activation、batch_size、epochs）。 |
| **DEM** | `train.loc[:, 'sex_M':'age_group_5']` | 仅人口统计学。 |
| **Molecular** | `train.loc[:, 'molecular_weight':'covalent_unit_count']` | 仅分子描述符。 |
| **Chemical** | `train.loc[:, "1":"880"]` | 仅化学亚结构指纹。 |
| **Bio** | 生物特征列（DrugBank 蛋白） | 仅生物特征。 |
| **BIO+DEMO** | 生物 + `sex_M`~`age_group_5` | 生物 + 人口统计。 |
| **Chemical + demo** | 化学 + 人口统计 | 化学指纹 + 人口统计。 |
| **Molecular +Demo** | 分子 + 人口统计 | 分子描述符 + 人口统计。 |
| **Molecular +bio** | 分子 + 生物 | 分子 + 生物特征。 |
| **Chemical molecular** | 分子 + 化学 | 分子描述符 + 化学指纹。 |
| **Chemical Bio** | 化学 + 生物 | 化学 + 生物特征。 |
| **Chemical + Molecular + Demographic** | 化学 + 分子 + 人口统计 | 三组特征 + 人口统计。 |
| **bio + Molecular + Demographic** | 生物 + 分子 + 人口统计 | 生物 + 分子 + 人口统计。 |
| **bio chemical demo** | 生物 + 化学 + 人口统计 | 三组特征 + 人口统计。 |
| **molecular, chemical, and bio** | 分子 + 化学 + 生物 | 三组特征，无人口统计。 |

---

## 4. Grid Search（部分小节）

| 代码/步骤 | 作用 |
|-----------|------|
| `def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):` | 定义可配置的 Keras 模型，供 GridSearchCV 调参。 |
| `KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)` | 用 KerasClassifier 包装成 sklearn 兼容的估计器。 |
| `param_grid = { 'optimizer': [...], 'dropout_rate': [...], ... }` | 搜索空间：优化器、Dropout、激活函数、batch_size、epochs。 |
| `GridSearchCV(..., scoring='roc_auc', cv=3)` | 3 折交叉验证，以 ROC-AUC 为评分。 |
| `grid_result.best_estimator_.model` | 得到最优 Keras 模型用于测试集评估。 |

---

## 5. 小结

| 阶段 | 主要动作 |
|------|----------|
| 数据与划分 | 读入 `data.csv`，分子 K-means 25 簇，按簇 80/20 划分 train/test，打 `type`。 |
| 预处理 | 分子特征 Normalizer 归一化，删除 sex_F、age_group_1，拆出 train/test。 |
| 建模 | 多种特征组合 × 同一 DL 结构（512→Dropout→256→30 sigmoid），binary_crossentropy + AUC。 |
| 评估 | 每类 AUC/MAP、宏平均 AUC、宏平均 MAP；部分小节含 Grid Search。 |

整体流程与 Farnoush 文章总结中的 DL 设计一致：4 层结构、30 个输出节点、Sigmoid、交叉熵、AUC/MAP 评估，并与 Modeling 1-RF 共用同一数据与划分策略，便于与 RF 结果对比。

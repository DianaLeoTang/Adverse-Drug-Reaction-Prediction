# Modeling 1-RF 代码分析

本文档对应 `Modeling 1-RF.ipynb` / `modeling_1_rf.py`，按执行顺序说明各段代码的作用。

---

## 1. 数据加载与初步清洗

| 代码/步骤 | 作用 |
|-----------|------|
| `pd.read_csv("data.csv")` | 读取合并后的主表：行=样本（药物/患者相关记录），列=化学特征 + 分子描述符 + 生物特征 + 人口统计 + 30 种 ADR 标签。 |
| `data.loc[:, "A0A023W3H0":"W7JWW5"]` | 取出生物特征列范围，用于检查全零列。 |
| `[col for col in data2.columns if data2[col].sum() == 0]` | 找出列和为 0 的列（无变异，对建模无贡献）。 |
| `data.loc[:, (data != 0).any(axis=0)]` | **删除全零列**，只保留至少有一个非零值的列，减小维度、避免无效特征。 |
| `data.loc[:, "molecular_weight":"covalent_unit_count"]` | 提取 **分子描述符子表**（PubChem 17 维），用于后续聚类和特征组合。 |

---

## 2. 聚类数选择（可选）

| 代码/步骤 | 作用 |
|-----------|------|
| `silhouette_score(molecular_df, labels)` 循环 | 对簇数 2~29 计算 **轮廓系数**，评估聚类紧密度与分离度，辅助选 K。 |
| `plt.plot(range(2, 30), silhouette_scores)` | 绘制轮廓系数随簇数变化曲线。 |
| `kmeans.inertia_` 循环 | **肘部法**：对簇数 1~29 计算 WCSS（簇内平方和），画图辅助确定“拐点”作为 K。 |
| `plt.plot(range(1, 30), wcss)` | 绘制肘部曲线。 |

---

## 3. 按分子簇划分训练/测试（防泄漏）

| 代码/步骤 | 作用 |
|-----------|------|
| `KMeans(n_clusters=25, random_state=42).fit(molecular_df)` | 在 **分子特征** 上做 K-means，得到 **25 个分子簇**（与文章设计一致）。 |
| `cluster_labels.value_counts() / sum` | 计算每个簇的样本占比。 |
| `tmp.sample(frac=1, random_state=30).cumsum()` | 随机打乱簇顺序后做累加占比，用于按比例划分簇。 |
| `train_clusters = set(tmp[lambda x: x<.80].index)` | 累计占比 &lt; 80% 的簇归为 **训练簇**。 |
| `test_clusters = set(tmp[lambda x: x>=.80].index)` | 累计占比 ≥ 80% 的簇归为 **测试簇**。 |
| `train_mol` / `test_mol` | 根据簇标签得到训练/测试对应的 **分子子表索引**（行索引）。 |
| `data.loc[:, "molecular_weight":"covalent_unit_count"].isin(train_mol).all(axis=1)` | 判断每条样本的分子特征行是否属于 `train_mol`，从而给该样本打 `type='train'` 或 `'test'`。 |

**目的**：在 **分子类别层面** 做 80/20 划分，同一分子类别不会同时出现在训练集和测试集中，避免信息泄漏。

---

## 4. 分子特征标准化与冗余列删除

| 代码/步骤 | 作用 |
|-----------|------|
| `Normalizer()` + `fit_transform(...)` | 对 **分子描述符**（`molecular_weight` ~ `covalent_unit_count`）做行向量的 L2 归一化。 |
| `data.drop(["sex_F", "age_group_1"], axis=1, inplace=True)` | 删除与保留列共线或冗余的人口统计列（如 sex_F 与 sex_M 二选一、age_group_1 与其余年龄组冗余）。 |
| `train = data[data["type"] == "train"]` | 按 `type` 拆出 **训练集**。 |
| `test = data[data["type"] == "test"]` | 按 `type` 拆出 **测试集**。 |

---

## 5. 随机森林多标签实验（不同特征组合）

整体流程（各小节共用）：

1. 从 `train` / `test` 中按列范围取 **特征 X** 和 **标签 Y**（`cardiac failure` ~ `vomiting`，共 30 个 ADR）。
2. 对 **每个 ADR 标签** 单独训练一个二分类随机森林，预测概率。
3. 计算该标签的 **AUC**（`roc_auc_score`）和 **MAP**（`average_precision_score`）。
4. 累加各标签的 **特征重要性**（`clf.feature_importances_`），再除以标签数得到平均重要性。
5. 汇报 **平均 AUC、平均 MAP**，以及 **Top 20 重要特征名称**（`np.argsort(feature_importances)[-20:]`）。

各小节仅 **特征 X 的列选取** 不同：

| 小节 | 特征选取 | 含义 |
|------|----------|------|
| **ALL** | `train.loc[:, 'sex_M':'868']` | 全部特征：人口统计 + 分子 + 化学(0~868) + 生物；RF 300 棵树。 |
| **Dem** | `train.loc[:, 'sex_M':'age_group_5']` | 仅人口统计学；RF 100 棵树。 |
| **Molecular** | `train.loc[:, 'molecular_weight':'covalent_unit_count']` | 仅分子描述符；RF 200 棵树。 |
| **Chemical** | `train.loc[:, "0":"868"]` | 仅化学亚结构指纹；RF 200 棵树。 |
| **BIO** | `train.loc[:, "A0A024R8I1":"Q9Y6Y9"]` | 仅生物特征（DrugBank 蛋白）；RF 300 棵树。 |
| **BIO DEMO** | `concat([A0A024R8I1:Q9Y6Y9], [sex_M:age_group_5])` | 生物 + 人口统计；RF 300 棵树。 |
| **Chemical + demo** | `concat(["0":"868"], [sex_M:age_group_5])` | 化学 + 人口统计；RF 200 棵树。 |
| **Molecular +Demo** | `concat([molecular_weight:covalent_unit_count], [sex_M:age_group_5])` | 分子 + 人口统计；含 GridSearchCV 调参或固定 1000 棵树等变体。 |
| **Molecular +bio** | 分子 + 生物 | 分子与生物特征组合。 |
| **Chemical molecular** | `concat([molecular_weight:covalent_unit_count], ["0":"868"])` | 分子 + 化学。 |
| **Chemical Bio** | `concat([A0A024R8I1:Q9Y6Y9], ["0":"868"])` | 化学 + 生物。 |
| **Chemical + Molecular + Demographic** | `concat([molecular_weight:"868"], [sex_M:age_group_5])` | 化学 + 分子 + 人口统计。 |
| **bio + Molecular + Demographic** | `train.loc[:, "sex_M":"covalent_unit_count"]` | 生物 + 分子 + 人口统计。 |
| **bio chemical demo** | 生物 + 化学 + 人口统计 | 三组特征 + 人口统计。 |
| **molecular, chemical, and bio** | 分子 + 化学 + 生物 | 三组特征无人口统计。 |

---

## 6. 特征重要性与 Top 20

| 代码/步骤 | 作用 |
|-----------|------|
| `feature_importances += clf.feature_importances_`（循环 30 个标签） | 累加每个 ADR 二分类器的特征重要性。 |
| `feature_importances /= len(y_train.columns)` | 除以 30，得到 **跨标签平均特征重要性**。 |
| `np.argsort(feature_importances)[-20:]` | 取重要性排名 **前 20** 的特征索引。 |
| `x_train.columns[top_20_feature_indices]` | 得到 **Top 20 特征名称**，用于可解释性分析（与文章“5 人口统计 + 13 分子 + 2 生物”对应）。 |

Notebook 中仅计算并展示 Top 20 名称/重要性；若需复现文章中的 **精简模型**（仅用 Top 20 特征再训练并报 AUC），需在得到 `top_20_feature_names` 后，用这 20 列重新构造 `x_train`/`x_test` 再训练一次 RF 并计算平均 AUC/MAP。

---

## 7. 后续小节（Feature Importance RF / DL / top 20 USING DL 等）

- **Feature Importance RF**：在某一特征组合下，对特征重要性做更细的分析或可视化。
- **each feature individually**：可能对单个特征或单类特征做单独建模/消融。
- **top 20 USING DL**：用 **深度学习** 模型（如多标签 MLP）做预测，并基于梯度或其它方式得到“重要特征”，与 RF 的 Top 20 对比或组合。

这些部分在 `modeling_1_rf.py` 中保留为与 notebook 一致的代码块；详细行为需结合 notebook 中对应 cell 的输入输出查看。

---

## 8. 小结

| 阶段 | 主要动作 |
|------|----------|
| 数据 | 读入 `data.csv`，删全零列，提取分子子表。 |
| 划分 | 分子特征 K-means 25 簇，按簇 80/20 划分 train/test，再按行打 `type`。 |
| 预处理 | 分子特征 Normalizer 归一化，删除 sex_F、age_group_1。 |
| 建模 | 多种特征组合 × 30 个 ADR 二分类 RF，汇报平均 AUC、MAP 与 Top 20 特征。 |
| 评估 | AUC（roc_auc_score）、MAP（average_precision_score）。 |

整体流程与 Farnoush 文章总结中的设计一致：数据源（FAERS + DrugBank + PubChem）、30 种 ADR、K-means 25 簇 80/20 分割、随机森林多标签与特征重要性、多种特征组合实验及 AUC/MAP 评估。

"""
Modeling 1-RF: 基于随机森林的药品不良反应(ADR)多标签预测
来源: Modeling 1-RF.ipynb
数据: FAERS + DrugBank + PubChem, 30种ADR, 多类特征组合
"""

# 可选: 若在其它环境运行需切换工作目录，可取消下一行注释并修改路径
# import os; os.chdir("/path/to/Adverse-Drug-Reaction-Prediction")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict

# =============================================================================
# 1. 数据加载与初步清洗
# =============================================================================
# 读取合并后的特征与标签数据 (data.csv: 化学+分子+生物+人口统计特征, 30种ADR标签)
data = pd.read_csv("data.csv")

# 取生物特征列范围，用于检查全零列（后续会删除全零列以减小维度）
data2 = data.loc[:, "A0A023W3H0":"W7JWW5"]
columns_with_zero_sum = [col for col in data2.columns if data2[col].sum() == 0]
len(columns_with_zero_sum)
columns_with_zero_sum

# 删除所有列和为0的列，保留至少有一个非零值的列
data = data.loc[:, (data != 0).any(axis=0)]
data.shape

# 可选: 查看所有列名
# for i in data.columns:
#     print(i)

# 提取分子描述符子表 (PubChem 17维: molecular_weight ~ covalent_unit_count)，用于聚类与后续特征组合
molecular_df = data.loc[:, "molecular_weight":"covalent_unit_count"]

# =============================================================================
# 2. 聚类数选择（可选：肘部法 / 轮廓系数）
# =============================================================================
silhouette_scores = []
# 轮廓系数: 评估不同簇数(2~29)的聚类质量，用于辅助选择 K=25
for i in range(2, 30):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(molecular_df)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(molecular_df, labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores
plt.plot(range(2, 30), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# 肘部法: 绘制簇数 1~29 的 WCSS，辅助确定 K
wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(molecular_df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 30), wcss)
plt.show()

# =============================================================================
# 3. 按分子簇 80/20 划分训练/测试（防止同一分子类别同时出现在训练与测试中）
# =============================================================================
kmeans = KMeans(n_clusters=25, random_state=42).fit(molecular_df)
cluster_labels = pd.Series(kmeans.labels_)
# 各簇样本占比 -> 随机打乱簇顺序 -> 累加占比，前 80% 的簇为训练簇，后 20% 为测试簇
tmp = cluster_labels.value_counts()
tmp = tmp / tmp.sum()
tmp = tmp.sample(frac=1, random_state=30).cumsum()
train_clusters = set(tmp[lambda x: x < .80].index)
test_clusters = set(tmp[lambda x: x >= .80].index)
train_mol = molecular_df.iloc[cluster_labels[lambda s: s.map(lambda x: x in train_clusters)].index]
test_mol = molecular_df.iloc[cluster_labels[lambda s: s.map(lambda x: x in test_clusters)].index]
train_test_split = pd.concat([
    train_mol[[]].reset_index().assign(type="train"),
    test_mol[[]].reset_index().assign(type="test"),
])

# 根据分子特征行是否属于 train_mol/test_mol，为每条样本打上 type = train / test
data['type'] = 'Neither'
mask_B = data.loc[:, "molecular_weight": "covalent_unit_count"].isin(train_mol).all(axis=1)
data.loc[mask_B, 'type'] = 'train'
mask_C = data.loc[:, "molecular_weight": "covalent_unit_count"].isin(test_mol).all(axis=1)
data.loc[mask_C, 'type'] = 'test'
data["type"].unique()

# =============================================================================
# 4. 分子特征标准化与冗余列删除
# =============================================================================
scaler = Normalizer()
data.loc[:, "molecular_weight":"covalent_unit_count"] = scaler.fit_transform(
    data.loc[:, "molecular_weight":"covalent_unit_count"]
)
# 删除与保留列共线的人口统计列（如 sex_F 与 sex_M 二选一、age_group_1 与其余年龄组冗余）
data.drop(["sex_F", "age_group_1"], axis=1, inplace=True)

# 按 type 拆分为训练集与测试集
train = data[data["type"] == "train"]
test = data[data["type"] == "test"]
# 可选: 检查测试集占比
# test.shape[0] / data.shape[0]


# =============================================================================
# 5. 随机森林多标签实验：不同特征组合
# =============================================================================
# 以下各小节：取不同特征子集 X，对 30 个 ADR 标签分别训练 RF，汇报平均 AUC、MAP 与 Top 20 特征重要性。

# -----------------------------------------------------------------------------
# ALL: 全部特征（人口统计 sex_M~age_group_5 + 分子 + 化学 0~868 + 生物 A0A024R8I1~Q9Y6Y9）
# -----------------------------------------------------------------------------
x_train = train.loc[:, 'sex_M':'868']
x_test = test.loc[:, 'sex_M':'868']
y_train = train.loc[:, 'cardiac failure':'vomiting']
y_test = test.loc[:, 'cardiac failure':'vomiting']

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-30:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')

top_20_feature_indices = np.argsort(feature_importances)[-21:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')

np.sort(feature_importances)[-20:]

x_train

feature_importances[-20:]

p10635_importance = feature_importances[x_train.columns.get_loc('P10635')]
p10635_importance


# -----------------------------------------------------------------------------
# Dem: 仅人口统计学特征 (sex_M ~ age_group_5)
# -----------------------------------------------------------------------------
x_train = train.loc[:, 'sex_M': 'age_group_5']
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'sex_M': 'age_group_5']
y_test = test.loc[:, 'cardiac failure': "vomiting"]
# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


# -----------------------------------------------------------------------------
# Molecular: 仅分子描述符 (molecular_weight ~ covalent_unit_count)
# -----------------------------------------------------------------------------
x_train = train.loc[:, 'molecular_weight': 'covalent_unit_count']
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'molecular_weight': 'covalent_unit_count']
y_test = test.loc[:, 'cardiac failure': "vomiting"]
# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


# -----------------------------------------------------------------------------
# Chemical: 仅化学亚结构指纹 (列 "0" ~ "868")
# -----------------------------------------------------------------------------
x_train = train.loc[:, "0":"868"]
y_train = train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "0":"868"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]
# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


# -----------------------------------------------------------------------------
# BIO: 仅生物特征 (DrugBank 蛋白列 A0A024R8I1 ~ Q9Y6Y9)
# -----------------------------------------------------------------------------
x_train = train.loc[:, "A0A024R8I1":"Q9Y6Y9"]
# x_train.shape

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
# Combine the relevant features from the training and testing datasets
x_train= train.loc[:, "A0A024R8I1":"Q9Y6Y9"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "A0A024R8I1":"Q9Y6Y9"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')

best_model


#==============================================================================
# BIO DEMO
#==============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict

# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, "A0A024R8I1":"Q9Y6Y9"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train = train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "A0A024R8I1":"Q9Y6Y9"], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')

best_model


#==============================================================================
# Chemical + demo
#==============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict

# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, "0":"868"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "0":"868"], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# Molecular +Demo
#==============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict

# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

auc_scores = []
map_scores = []
feature_importances = np.zeros((y_train.shape[1], x_train.shape[1]))

# Define the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 200, 800],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

for i in range(y_train.shape[1]):
    # Extract the label column
    y_train_label = y_train.iloc[:, i]
    y_test_label = y_test.iloc[:, i]
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train_label)
    
    # Train the best model from grid search on the entire training data
    best_model = grid_search.best_estimator_
    best_model.fit(x_train, y_train_label)
    
    # Predict probabilities for the positive class
    y_prob = best_model.predict_proba(x_test)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_test_label, y_prob)
    auc_scores.append(auc)
    
    # Calculate Mean Average Precision (MAP)
    map_score = average_precision_score(y_test_label, y_prob)
    map_scores.append(map_score)
    
    # Get feature importances and add to the total for this target variable
    feature_importances[i, :] = best_model.feature_importances_

# Calculate the average AUC and MAP across all target variables
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate the average feature importances across all target variables
average_feature_importances = np.mean(feature_importances, axis=0)

# Print the average AUC and MAP
print(f"\nAverage AUC across all target variables: {average_auc:.4f}")
print(f"Average MAP across all target variables: {average_map:.4f}")

# Identify the top 10 most important features
sorted_indices = np.argsort(average_feature_importances)[::-1]
top_10_features = sorted_indices[:10]

# Print the top 10 most important features
print("\nTop 10 Most Important Features:")
for idx in top_10_features:
    print(f"{x_train.columns[idx]}: {average_feature_importances[idx]:.4f}")

from sklearn.ensemble import RandomForestClassifier

# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=1000, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# Molecular +bio
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "A0A024R8I1":"Q9Y6Y9"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "A0A024R8I1":"Q9Y6Y9"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# Chemical molecular
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "0":"868"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "0":"868"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# Chemical Bio
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, "A0A024R8I1":"Q9Y6Y9"], train.loc[:, "0":"868"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "A0A024R8I1":"Q9Y6Y9"], test.loc[:, "0":"868"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=400, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# Chemical + Molecular + Demographic
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = pd.concat([train.loc[:, "molecular_weight":"868"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "molecular_weight":"868"], test.loc[:, "sex_M":"age_group_5"]],  axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# bio + Molecular + Demographic
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = train.loc[:, "sex_M":"covalent_unit_count"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "sex_M":"covalent_unit_count"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# bio chemical demo
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = pd.concat([ train.loc[:, "sex_M":"Q9Y6Y9"], train.loc[:, "0":"868"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([ test.loc[:, "sex_M":"Q9Y6Y9"], test.loc[:, "0":"868"]],  axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Placeholder for the metrics and feature importances
auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# molecular, chemical, and bio
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = train.loc[:, "A0A024R8I1":"868"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "A0A024R8I1":"868"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=40)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# Feature Importance RF
#==============================================================================

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = train[['P02768', 'P08183', 'undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
       'age_group_5', 'age_group_4', 'sex_M']]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test[[
       'P02768', 'P08183', 'undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
       'age_group_5', 'age_group_4', 'sex_M']]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=400, random_state=42)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = train[['P02768', 'P08183', 'undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp']]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test[[
       'P02768', 'P08183', 'undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp']]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=400, random_state=42)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')

from sklearn.ensemble import RandomForestClassifier


# Combine the relevant features from the training and testing datasets
x_train = train[['P02768', 'P08183']]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test[[
       'P02768', 'P08183']]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

auc_scores = []
map_scores = []
feature_importances = np.zeros(x_train.shape[1])

# Loop through each label
for label in y_train.columns:
    y_train_label = y_train[label]
    y_test_label = y_test[label]

    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=400, random_state=42)
    clf.fit(x_train, y_train_label)

    # Predict probabilities
    y_pred_prob = clf.predict_proba(x_test)[:, 1]

    # Calculate AUC and MAP
    auc = roc_auc_score(y_test_label, y_pred_prob)
    map_score = average_precision_score(y_test_label, y_pred_prob)

    # Append the scores
    auc_scores.append(auc)
    map_scores.append(map_score)

    # Accumulate feature importances
    feature_importances += clf.feature_importances_

# Calculate average AUC and MAP
average_auc = np.mean(auc_scores)
average_map = np.mean(map_scores)

# Calculate average feature importances across all labels
feature_importances /= len(y_train.columns)

# Calculate average of the top 20 feature importances
top_20_feature_importances = np.sort(feature_importances)[-20:]

# Display the results
print(f'Average AUC: {average_auc}')
print(f'Average MAP: {average_map}')
print(f'Average of Top 20 Feature Importances: {np.mean(top_20_feature_importances)}')

# Optional: if you want to display the names of the top 20 features
top_20_feature_indices = np.argsort(feature_importances)[-20:]
top_20_feature_names = x_train.columns[top_20_feature_indices]
print(f'Top 20 Features: {top_20_feature_names}')


#==============================================================================
# each feature individuially
#==============================================================================

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define the features and labels for training and testing
x_train = train[['P02768', 'P08183', 'undefined_atom_stereo_count', 'defined_atom_stereo_count',
                 'h_bond_donor_count', 'complexity', 'exact_mass', 'rotatable_bond_count',
                 'heavy_atom_count', 'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count',
                 'tpsa', 'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
                 'age_group_5', 'age_group_4', 'sex_M']]
y_train = train.loc[:, 'cardiac failure': 'vomiting']
x_test = test[['P02768', 'P08183', 'undefined_atom_stereo_count', 'defined_atom_stereo_count',
               'h_bond_donor_count', 'complexity', 'exact_mass', 'rotatable_bond_count',
               'heavy_atom_count', 'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count',
               'tpsa', 'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
               'age_group_5', 'age_group_4', 'sex_M']]
y_test = test.loc[:, 'cardiac failure': 'vomiting']

# Initialize lists to store the AUC and MAP scores for each feature
feature_auc_scores = []
feature_map_scores = []

# Loop through each feature
for feature in x_train.columns:
    auc_scores = []
    map_scores = []
    
    # Loop through each label
    for label in y_train.columns:
        y_train_label = y_train[label]
        y_test_label = y_test[label]
        
        # Use only the current feature for training and testing
        x_train_feature = x_train[[feature]]
        x_test_feature = x_test[[feature]]
        
        # Initialize and train the random forest classifier
        clf = RandomForestClassifier(n_estimators=10, random_state=60)
        clf.fit(x_train_feature, y_train_label)
        
        # Predict probabilities
        y_pred_prob = clf.predict_proba(x_test_feature)[:, 1]
        
        # Calculate AUC and MAP
        auc = roc_auc_score(y_test_label, y_pred_prob)
        map_score = average_precision_score(y_test_label, y_pred_prob)
        
        # Append the scores
        auc_scores.append(auc)
        map_scores.append(map_score)
    
    # Calculate average AUC and MAP for the current feature
    feature_auc_scores.append(np.mean(auc_scores))
    feature_map_scores.append(np.mean(map_scores))

# Display the results
for feature, auc, map_score in zip(x_train.columns, feature_auc_scores, feature_map_scores):
    print(f'Feature: {feature} - Average AUC: {auc} - Average MAP: {map_score}')


#==============================================================================
# top 20 USING DL
#==============================================================================

from sklearn.metrics import roc_auc_score, average_precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import random
import tensorflow as tf

x_train = train[['P02768', 'P08183','undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
       'age_group_5', 'age_group_4', 'sex_M']]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test[['P02768', 'P08183','undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
       'age_group_5', 'age_group_4', 'sex_M']]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.2)


predictions = model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate macro average AUC
macro_auc = roc_auc_score(y_test, predictions, average='macro')
print(f"Macro AUC: {macro_auc}")

map_per_class = average_precision_score(y_test, predictions, average=None)
print(f"MAP per class: {map_per_class}")

# Calculate macro average MAP
macro_map = average_precision_score(y_test, predictions, average='macro')
print(f"Macro MAP: {macro_map}")

from sklearn.metrics import roc_auc_score, average_precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import random
import tensorflow as tf

x_train = train[['P02768', 'P08183','undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp']]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test[['P02768', 'P08183','undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp']]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.2)


predictions = model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate macro average AUC
macro_auc = roc_auc_score(y_test, predictions, average='macro')
print(f"Macro AUC: {macro_auc}")

map_per_class = average_precision_score(y_test, predictions, average=None)
print(f"MAP per class: {map_per_class}")

# Calculate macro average MAP
macro_map = average_precision_score(y_test, predictions, average='macro')
print(f"Macro MAP: {macro_map}")

from sklearn.metrics import roc_auc_score, average_precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import random
import tensorflow as tf

x_train = train[['undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
       'age_group_5', 'age_group_4', 'sex_M']]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test[['undefined_atom_stereo_count',
       'defined_atom_stereo_count', 'h_bond_donor_count', 'complexity',
       'exact_mass', 'rotatable_bond_count', 'heavy_atom_count',
       'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count', 'tpsa',
       'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
       'age_group_5', 'age_group_4', 'sex_M']]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2)


predictions = model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate macro average AUC
macro_auc = roc_auc_score(y_test, predictions, average='macro')
print(f"Macro AUC: {macro_auc}")

map_per_class = average_precision_score(y_test, predictions, average=None)
print(f"MAP per class: {map_per_class}")

# Calculate macro average MAP
macro_map = average_precision_score(y_test, predictions, average='macro')
print(f"Macro MAP: {macro_map}")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Set the random seed for reproducibility
np.random.seed(44)

# Define the features and labels for training and testing
x_train = train[['P02768', 'P08183', 'undefined_atom_stereo_count', 'defined_atom_stereo_count',
                 'h_bond_donor_count', 'complexity', 'exact_mass', 'rotatable_bond_count',
                 'heavy_atom_count', 'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count',
                 'tpsa', 'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
                 'age_group_5', 'age_group_4', 'sex_M']]
y_train = train.loc[:, 'cardiac failure':'vomiting']
x_test = test[['P02768', 'P08183', 'undefined_atom_stereo_count', 'defined_atom_stereo_count',
               'h_bond_donor_count', 'complexity', 'exact_mass', 'rotatable_bond_count',
               'heavy_atom_count', 'molecular_weight', 'monoisotopic_mass', 'covalent_unit_count',
               'tpsa', 'h_bond_acceptor_count', 'xlogp', 'age_group_2', 'age_group_3',
               'age_group_5', 'age_group_4', 'sex_M']]
y_test = test.loc[:, 'cardiac failure':'vomiting']

# Initialize lists to store the AUC and MAP scores for each feature
feature_auc_scores = []
feature_map_scores = []

# Define a function to create the model
def create_model(input_shape):
    model = Sequential([
        Dense(512, input_dim=input_shape, activation='relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

# Loop through each feature
for feature in x_train.columns:
    auc_scores = []
    map_scores = []
    
    # Loop through each label
    for label in y_train.columns:
        y_train_label = y_train[label]
        y_test_label = y_test[label]
        
        # Use only the current feature for training and testing
        x_train_feature = x_train[[feature]]
        x_test_feature = x_test[[feature]]
        
        # Create and train the neural network model
        model = create_model(input_shape=x_train_feature.shape[1])
        model.fit(x_train_feature, y_train_label, epochs=10, batch_size=32, verbose=0)
        
        # Predict probabilities
        y_pred_prob = model.predict(x_test_feature).flatten()
        
        # Calculate AUC and MAP
        auc = roc_auc_score(y_test_label, y_pred_prob)
        map_score = average_precision_score(y_test_label, y_pred_prob)
        
        # Append the scores
        auc_scores.append(auc)
        map_scores.append(map_score)
    
    # Calculate average AUC and MAP for the current feature
    feature_auc_scores.append(np.mean(auc_scores))
    feature_map_scores.append(np.mean(map_scores))

# Display the results
for feature, auc, map_score in zip(x_train.columns, feature_auc_scores, feature_map_scores):
    print(f'Feature: {feature} - Average AUC: {auc} - Average MAP: {map_score}')

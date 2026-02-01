"""
Modeling 2-DL: 基于深度学习的药品不良反应(ADR)多标签预测
来源: Modeling 2-DL.ipynb
数据: 与 Modeling 1-RF 相同 (data.csv)，划分方式一致
模型: Keras Sequential, 输入层 + 2 隐藏层(512/256) + 输出层(30 节点 Sigmoid), 交叉熵 + AUC
"""

# import os; os.chdir("/path/to/Adverse-Drug-Reaction-Prediction")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# =============================================================================
# 1. 数据加载与按分子簇划分（与 Modeling 1-RF 一致）
# =============================================================================
data = pd.read_csv("data.csv")
molecular_df = data.loc[:, "molecular_weight":"covalent_unit_count"]

# K-means 25 簇，按簇 80/20 划分训练/测试
kmeans = KMeans(n_clusters=25, random_state=42).fit(molecular_df)
cluster_labels = pd.Series(kmeans.labels_)
tmp = cluster_labels.value_counts()
tmp = tmp / tmp.sum()
tmp = tmp.sample(frac=1, random_state=30).cumsum()
train_clusters = set(tmp[lambda x: x<.80].index)
test_clusters = set(tmp[lambda x: x>=.80].index)
train_mol = molecular_df.iloc[cluster_labels[lambda s: s.map(lambda x: x in train_clusters)].index]
test_mol = molecular_df.iloc[cluster_labels[lambda s: s.map(lambda x: x in test_clusters)].index]

train_test_split = pd.concat([
    train_mol[[]].reset_index().assign(type="train"),
    test_mol[[]].reset_index().assign(type="test"),
])

data['type'] = 'Neither'

# Check if the values in DataFrame A exist in DataFrame B
mask_B = data.loc[:, "molecular_weight": "covalent_unit_count"].isin(train_mol).all(axis=1)
data.loc[mask_B, 'type'] = 'train'

mask_C = data.loc[:, "molecular_weight": "covalent_unit_count"].isin(test_mol).all(axis=1)
data.loc[mask_C, 'type'] = 'test'

# data["type"].unique()

# =============================================================================
# 2. 分子特征标准化与冗余列删除，拆分为 train / test
# =============================================================================
scaler = Normalizer()
data.loc[:, "molecular_weight":"covalent_unit_count"] = scaler.fit_transform(
    data.loc[:, "molecular_weight":"covalent_unit_count"]
)
data.drop(["sex_F", "age_group_1"], axis=1, inplace=True)
train = data[data["type"] == "train"]
test = data[data["type"] == "test"]
# test.shape[0] / data.shape[0]

# =============================================================================
# 3. 深度学习多标签实验：不同特征组合
# =============================================================================
# 网络结构: Dense(512) -> Dropout -> Dense(256) -> Dense(30, sigmoid)；损失 binary_crossentropy，指标 AUC/MAP

# -----------------------------------------------------------------------------
# ALL: 全部特征 (sex_M ~ "880")
# -----------------------------------------------------------------------------
# 可选: Grid Search 调参 (optimizer, dropout_rate, activation, batch_size, epochs)

x_train = train.loc[:, 'sex_M':"880"]
y_train = train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'sex_M':"880"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train = train.loc[:, 'sex_M':"880"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'sex_M':"880"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

from sklearn.metrics import roc_auc_score

predictions = model.predict(x_test)

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
# probabilities  # notebook 中查看预测概率用，运行时可注释

# -----------------------------------------------------------------------------
# DEM: 仅人口统计学特征 (sex_M ~ age_group_5)
# -----------------------------------------------------------------------------

x_train= train.loc[:, 'sex_M': 'age_group_5']
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'sex_M': 'age_group_5']
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train= train.loc[:, 'sex_M': 'age_group_5']
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'sex_M': 'age_group_5']
y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='tanh', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='tanh'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Molecular: 仅分子描述符 (molecular_weight ~ covalent_unit_count)
# -----------------------------------------------------------------------------

x_train= train.loc[:, 'molecular_weight': 'covalent_unit_count']
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'molecular_weight': 'covalent_unit_count']
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train= train.loc[:, 'molecular_weight': 'covalent_unit_count']
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, 'molecular_weight': 'covalent_unit_count']
y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Chemical: 仅化学亚结构指纹 ("1" ~ "880")
# -----------------------------------------------------------------------------

x_train= train.loc[:, "1":"880"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "1":"880"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]
# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train= train.loc[:, "1":"880"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "1":"880"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Bio: 仅生物特征 (DrugBank 蛋白列)
# -----------------------------------------------------------------------------

x_train= train.loc[:, "A0A068JFB7":"W7JWW5"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "A0A068JFB7":"W7JWW5"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]
# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train= train.loc[:, "A0A068JFB7":"W7JWW5"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "A0A068JFB7":"W7JWW5"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# BIO+DEMO: 生物特征 + 人口统计
# -----------------------------------------------------------------------------

x_train = pd.concat([train.loc[:, "A0A068JFB7":"W7JWW5"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "A0A068JFB7":"W7JWW5"], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train = pd.concat([train.loc[:, "A0A068JFB7":"W7JWW5"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "A0A068JFB7":"W7JWW5"], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Chemical + demo: 化学 + 人口统计
# -----------------------------------------------------------------------------

x_train = pd.concat([train.loc[:, "1":"880"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "1":"880"], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')



x_train = pd.concat([train.loc[:, "1":"880"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "1":"880"], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]


# Define the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Molecular +Demo: 分子 + 人口统计
# -----------------------------------------------------------------------------

x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')

x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "sex_M":"age_group_5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]


# Define the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Molecular +bio: 分子 + 生物
# -----------------------------------------------------------------------------

x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "A0A068JFB7":"W7JWW5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "A0A068JFB7":"W7JWW5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "A0A068JFB7":"W7JWW5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "A0A068JFB7":"W7JWW5"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]


# Define the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Chemical molecular: 化学 + 分子
# -----------------------------------------------------------------------------

x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "1":"880"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "1":"880"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')



x_train = pd.concat([train.loc[:, 'molecular_weight': 'covalent_unit_count'], train.loc[:, "1":"880"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, 'molecular_weight': 'covalent_unit_count'], test.loc[:, "1":"880"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]


# Define the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Chemical Bio: 化学 + 生物
# -----------------------------------------------------------------------------

x_train = pd.concat([train.loc[:, "A0A068JFB7":"W7JWW5"], train.loc[:, "1":"880"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "A0A068JFB7":"W7JWW5"], test.loc[:, "1":"880"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')



x_train = pd.concat([train.loc[:, "A0A068JFB7":"W7JWW5"], train.loc[:, "1":"880"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "A0A068JFB7":"W7JWW5"], test.loc[:, "1":"880"]], axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# Chemical + Molecular + Demographic: 化学 + 分子 + 人口统计
# -----------------------------------------------------------------------------

x_train = pd.concat([train.loc[:, "molecular_weight":"880"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "molecular_weight":"880"], test.loc[:, "sex_M":"age_group_5"]],  axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')



x_train = pd.concat([train.loc[:, "molecular_weight":"880"], train.loc[:, "sex_M":"age_group_5"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([test.loc[:, "molecular_weight":"880"], test.loc[:, "sex_M":"age_group_5"]],  axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# bio + Molecular + Demographic: 生物 + 分子 + 人口统计
# -----------------------------------------------------------------------------

x_train = train.loc[:, "sex_M":"covalent_unit_count"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "sex_M":"covalent_unit_count"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train = train.loc[:, "sex_M":"covalent_unit_count"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "sex_M":"covalent_unit_count"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.9))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# bio chemical demo: 生物 + 化学 + 人口统计
# -----------------------------------------------------------------------------

x_train = pd.concat([ train.loc[:, "sex_M":"W7JWW5"], train.loc[:, "1":"880"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([ test.loc[:, "sex_M":"W7JWW5"], test.loc[:, "1":"880"]],  axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]


# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')


x_train = pd.concat([ train.loc[:, "sex_M":"W7JWW5"], train.loc[:, "1":"880"]], axis=1)
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = pd.concat([ test.loc[:, "sex_M":"W7JWW5"], test.loc[:, "1":"880"]],  axis=1)
y_test = test.loc[:, 'cardiac failure': "vomiting"]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

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


# -----------------------------------------------------------------------------
# molecular, chemical, and bio: 分子 + 化学 + 生物（无人口统计）
# -----------------------------------------------------------------------------

x_train = train.loc[:, "A0A023W3H0":"880"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "A0A023W3H0":"880"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]


# Define the model
def create_model(optimizer='adam', dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(name='auc', multi_label=True)])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# Define the parameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 100]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best AUC score: {grid_result.best_score_}")

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model
predictions = best_model.predict(x_test)

# Calculate AUC for each class
auc_per_class = roc_auc_score(y_test, predictions, average=None)
print(f"AUC per class: {auc_per_class}")

# Calculate the average AUC across all classes
average_auc = np.mean(auc_per_class)
print(f'Average AUC: {average_auc}')



x_train = train.loc[:, "A0A023W3H0":"880"]
y_train= train.loc[:, 'cardiac failure': "vomiting"]
x_test = test.loc[:, "A0A023W3H0":"880"]
y_test = test.loc[:, 'cardiac failure': "vomiting"]

y_test = test.loc[:, 'cardiac failure': "vomiting"]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model with AUC as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

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

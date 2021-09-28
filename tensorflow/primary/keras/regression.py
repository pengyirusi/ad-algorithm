# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/9/28 20:09
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # 绘制矩阵图
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Auto MPG 数据集：data 气缸数 排量 马力 重量 label 燃油效率
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

# 使用 pandas 导入数据集
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values='?',  # 缺失值填充为'?'
                          comment='\t',
                          sep=' ',  # 分隔符
                          skipinitialspace=True)  # 跳过分隔符后的空格
dataset = raw_dataset.copy()
print(dataset.head())  # 查看数据


# 数据清洗
print(dataset.isna().sum())  # 查看 Nan 的数据个数
dataset = dataset.dropna()  # 去掉 Nan 的数据
print(dataset.isna().sum())  # 又看一眼 果然删了


origin = dataset.pop("Origin")
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.head())


# 建立训练集和测试集
# 随机选择了 80%
train_dataset = dataset.sample(frac=0.8, random_state=0)
'''
    n : int, optional
        Number of items from axis to return. Cannot be used with `frac`.
        Default = 1 if `frac` = None.
    frac : float, optional
        Fraction of axis items to return. Cannot be used with `n`.
'''
test_dataset = dataset.drop(train_dataset.index)
# drop：去掉 train_dataset 的坐标


# 查看数据统计图
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
             diag_kind="kde")
# kde 核密度曲线 数据分布图
# plt.show()  # 也用 plt 显示


# 查看数据统计表格
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats)


# 获得 label 'MPG' 就是要预测的值
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 数据标准化 (x-平均值)/标准差
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


norm_train_data = norm(train_dataset)
norm_test_data = norm(test_dataset)
print(norm_train_data.head())


# 构建模型
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
print(model.summary())










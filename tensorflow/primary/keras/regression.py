# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/9/28 20:09
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  # 绘制矩阵图
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Auto MPG 数据集：data 气缸数 排量 马力 重量 label 燃油效率
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# print(dataset_path)

# 使用 pandas 导入数据集
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values='?',  # 缺失值填充为'?'
                          comment='\t',
                          sep=' ',  # 分隔符
                          skipinitialspace=True)  # 跳过分隔符后的空格
dataset = raw_dataset.copy()
# print(dataset.head())  # 查看数据


# 数据清洗
# print(dataset.isna().sum())  # 查看 Nan 的数据个数
dataset = dataset.dropna()  # 去掉 Nan 的数据
# print(dataset.isna().sum())  # 又看一眼 果然删了


# category -> one-hot
origin = dataset.pop("Origin")
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0


# print(dataset.head())
# print(type(dataset))  # <class 'pandas.core.frame.DataFrame'>
# pd.set_option('display.max_columns', None)  # 显示所有列，解决显示不全的问题
# print(dataset[0:2])  # 前两排
# print(dataset.iloc[0, :])  # 第一行


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
# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
#              diag_kind="kde")
# kde 核密度曲线 数据分布图
# plt.show()  # 也用 plt 显示


# 查看数据统计表格
train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()
# print(train_stats)


# 获得 label 'MPG' 就是要预测的值
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 数据标准化 (x-平均值)/标准差
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


norm_train_data = norm(train_dataset)
norm_test_data = norm(test_dataset)
# print(norm_train_data.head())


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
# print(model.summary())


# 试用一下未训练的模型
example_batch = norm_train_data[:10]
example_result = model.predict(example_batch)
# print(np.array(example_result))


# 训练模型
# 通过打印点来显示训练速度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCH = 1  # 1000
history = model.fit(
    norm_train_data, train_labels,
    epochs=EPOCH, validation_split=0.2,  # 验证集比例
    verbose=0,  # 什么都不显示
    callbacks=[PrintDot()]  # 打点
)

# print(type(history))  # <class 'tensorflow.python.keras.callbacks.History'>
'''
    将事件记录到 History 对象中的回调
    此回调将自动应用于每个 Keras 模型，History 对象由模型的 fit 方法返回
    在历元结束后，在模型上设置 History 属性，这将确保设置的状态为最新状态
'''

# 使用 history 对象中存储的统计信息可视化模型的训练进度
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\n', hist.tail())


# 画图
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


# plot_history(history)


# 上图中发现训练多了反而效果变差了
model = build_model()
# EarlyStopping: Stop training when a monitored metric has stopped improving.
# patience: Number of epochs with no improvement after which training will be stopped.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
    norm_train_data, train_labels, epochs=EPOCH,
    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()]
)
# plot_history(history)


# 测试集评价模型
loss, mae, mse = model.evaluate(norm_test_data, test_labels, verbose=2)
print('mae: ', mae)
test_predictions = model.predict(norm_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()










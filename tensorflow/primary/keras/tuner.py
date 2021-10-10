# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/10 12:10
"""

# Keras Tuner 超参调节库
# pip install -q -U keras-tuner

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# 准备数据集
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

# 构建模型
def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    hp_units = hp.Int


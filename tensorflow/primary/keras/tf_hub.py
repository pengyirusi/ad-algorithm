# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/9/27 20:41
    tensorflow_hub 一个用于在一行代码中从 TFHub 加载训练模型的库
"""
import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("Dataset version: ", tfds.__version__)

# 下载 IMDB 数据集
# 二分类 电影评论 1-positive 0-negative
train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    split=('train[:60%]', 'train[60%:]', 'test'),  # 15000 10000 25000
    as_supervised=True  # 有监督模式
)

# 探索数据
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print(train_examples_batch)
# print(train_labels_batch)  # tf.Tensor([0 0 0 1 1 1 0 0 0 0], shape=(10,), dtype=int64)
# batch 函数得到的是 tensor，直接就能放到模型中训练

'''
    构建模型
    1. 如何表示文本
    2. 模型有多少层
    3. 每层有多少个隐层单元
'''

# 词向量嵌入层
embedding = "https://hub.tensorflow.google.cn/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# print(hub_layer(train_labels_batch[:3]))  # 瞟一眼

# 完整模型
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
# print(model.summary())

# 损失函数与优化器
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# binary_crossentropy 分类
# mean_squared_error 回归


# 训练模型
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# 评估模型
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))


# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/8 20:14
"""

# 安装依赖 pip install pyyaml h5py

import os
import tensorflow as tf
from tensorflow import keras

# 使用 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 构建模型
def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


model = create_model()
print(model.summary())


# 在训练期间保存模型，边训练边保存
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# 用 cp_callback 保存模型的权重
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

# train the model with the new callback
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # 通过 callback 训练

# 创建一个 Tensorflow checkpoint 文件集合，这些文件在每个 epoch 结束时更新
os.listdir(checkpoint_dir)

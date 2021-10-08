# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/8 22:13
"""

# HDF5 格式的模型

import tensorflow as tf
import saveandload as sal

# 保存
model = sal.create_model()
model.fit(sal.train_images, sal.train_labels, epochs=5)
model.save('my_model.h5')

# 加载
new_model = tf.keras.models.load_model('my_model.h5')
print(new_model.summary())
loss, acc = new_model.evaluate(sal.test_images, sal.test_labels, verbose=2)
print('Loaded model, accuracy: {:5.2f}%'.format(100 * acc))

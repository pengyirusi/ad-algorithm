# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/8 22:06
"""

import saveandload as sal
import tensorflow as tf

print('=====start=====')

# 从保存的 model 中加载
model = tf.keras.models.load_model('saved_model/my_model')
print(model.summary())

# 评估模型
loss, acc = model.evaluate(sal.test_images, sal.test_labels, verbose=2)
print('Loaded model accuracy: {:5.2f}%'.format(100 * acc))


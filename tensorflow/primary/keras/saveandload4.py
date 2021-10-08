# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/8 21:50
"""

import tensorflow as tf
import saveandload as sal
import saveandload3 as sal3

print('=====start=====')

latest = tf.train.latest_checkpoint(sal3.checkpoint_dir)
print(latest)
model = sal.create_model()
model.load_weights(latest)
loss, acc = model.evaluate(sal.test_images, sal.test_labels, verbose=2)
print("Loaded model accuracy: {:5.2f}%".format(100 * acc))

# 中间选一个 weight
epoch_path = "training_2\cp-0020.ckpt"
print(epoch_path)
model.load_weights(epoch_path)
loss, acc = model.evaluate(sal.test_images, sal.test_labels, verbose=2)
print("Loaded model accuracy: {:5.2f}%".format(100 * acc))

# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/8 21:09
"""

import saveandload as sal

print('=====start=====')

# 用训练好的权重构建模型
model = sal.create_model()

# 评价一下新模型
loss, acc = model.evaluate(sal.test_images, sal.test_labels, verbose=2)
print("New model accuracy: {:5.2f}%".format(100 * acc))

# 加载已训练的权重
model.load_weights(sal.checkpoint_path)

# 重新评价模型
loss, acc = model.evaluate(sal.test_images, sal.test_labels, verbose=2)
print("Loaded model accuracy: {:5.2f}%".format(100 * acc))

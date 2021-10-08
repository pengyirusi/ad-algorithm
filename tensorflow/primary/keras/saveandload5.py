# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/8 22:04
"""

import saveandload as sal

# 保存整个模型
model = sal.create_model()
model.fit(sal.train_images, sal.train_labels, epochs=10)
model.save('saved_model/my_model')

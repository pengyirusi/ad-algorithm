# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/10/8 21:39
"""

import tensorflow as tf
from tensorflow import keras
import os.path
import saveandload as sal

print('=====start=====')

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# 每 5 个 epoch 存一个 callback
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size
)

model = sal.create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(
    sal.train_images,
    sal.train_labels,
    epochs=50,
    batch_size=batch_size,
    callbacks=[cp_callback],
    validation_data=(sal.test_images, sal.test_labels),
    verbose=0
)

os.listdir(checkpoint_dir)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/9/24 22:05
"""
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# 准备数据
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 添加通道维度
# numpy 数组切片中 ... 等同于 :
x_train = x_train[..., tf.newaxis]
'''
    添加通道维度也可以：
    x_train = tf.expand_dims(x_train, -1)
'''
x_test = x_test[..., tf.newaxis]

# shuffle 打乱数据集
# 表示新数据集将从此数据集中采样的元素数
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)). \
    shuffle(buffer_size=10000).batch(32)
'''
    from_tensor_slices 函数把原来的数据集按 32 切片，每 32 条数据为 1 组
    
In: 
    print(type(train_ds))
    print(train_ds._batch_size)
    print(train_ds._input_dataset)
Out:
    <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>
    <ShuffleDataset shapes: ((28, 28, 1), ()), types: (tf.float64, tf.uint8)>
    tf.Tensor(32, shape=(), dtype=int64)
'''
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# 模型子类化
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    # 重写 Model 里的 call 方法
    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.flatten(x1)
        x3 = self.d1(x2)
        output = self.d2(x3)
        return output


model = MyModel()

# 损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# 优化器
optimizer = tf.keras.optimizers.Adam()

# 训练指标 loss 和 accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

# 测试指标
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


# 训练模型
@tf.function  # 图像加速训练
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    # 测试模型
    for test_images, test_labels in test_ds:
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

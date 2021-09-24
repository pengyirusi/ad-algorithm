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
x_test = x_test[..., tf.newaxis]

# shuffle
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)). \
    shuffle(buffer_size=10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# 模型子类化
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# 损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# 优化器
optimizer = tf.keras.optimizers.Adam()





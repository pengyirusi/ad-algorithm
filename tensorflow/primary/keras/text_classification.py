# -*- coding: utf-8 -*-
"""
    @Author : weiyupeng
    @Time : 2021/9/26 21:32
"""
from tensorflow import keras

# 导入数据集
# 二分类 电影评论 1-positive 0-negative
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
'''
参数 num_words=10000 保留了训练数据中最常出现的 10000 个单词
为了保持数据规模的可管理性，低频词将被丢弃
所有的单词都被 1-10000 的数字代替了
'''

# # 了解数据集
# print(len(train_data), len(test_data))  # 25000 25000
# print(train_data[0])  # 整数数组
# print(len(train_data[0]))  # 218

# 整数转换回单词
word_index = imdb.get_word_index()
# print(type(word_index))  # <class 'dict'>

# 保留第一个索引
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

index_word = dict([(value, key) for (key, value) in word_index.items()])


# 数字转成文案
def number2text(number_list):
    return ' '.join([index_word[i] for i in number_list])


# print(number2text(train_data[0]))


# 预处理
# 每个文本的单词数量不同，需要凑成相同长度的向量，不够的直接补 0
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# 构建模型
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))  # 两个参数 input_dim output_dim
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
print(model.summary())

'''
    中间的两层 隐层单元
    隐层单元多：能学到更多特征，但是计算成本更高，且容易 overfit
'''

# 编译模型
model.compile(optimizer='adam',  # 优化器
              loss='binary_crossentropy',  # 损失函数 交叉熵
              metrics=['accuracy'])  # 指标

# 验证集
# 在训练根据验证集的结果对模型进行调整
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)  # 0=无提示，1=进度条，2=每个epoch一行

# 评估模型
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

history_dict = history.history
history_dict.keys()
print(
    history_dict)  # {'loss': [0.6920368075370789], 'accuracy': [0.5683333277702332], 'val_loss': [0.6900328397750854], 'val_accuracy': [0.6891999840736389]}

# 画一个准确率和损失值随时间变化的折线图
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')  # bo = blue-o
plt.plot(epochs, val_acc, 'b', label='Validation acc')  # b = blue-line
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 路透社新闻分类问题，将新闻分为46个不同的主题。
# 与 IMDB 评论一样，每个样本都是一个整数列表（表示单词索引）;标签是一个0-45的整数，对应46个话题
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)


# 将整数序列编码为二进制矩阵
# 举个例子，序列 [3, 5] 将会被转换为 10 000 维向量，只有索引为 3 和 5 的元素是 1，其余元素都是 0。
def vectorize_sequences(sequeces, dimension=10000):
    results = np.zeros((len(sequeces), dimension))  # 创建一个0矩阵
    for i, sequence in enumerate(sequeces):
        results[i, sequence] = 1
    return results


# 将训练数据和测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 使用keras内置方法将标签向量化，将array变为one-hot编码
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 定义模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
# 这里使用softmax激活函数，输出在46个类别上的概率分布
model.add(layers.Dense(46, activation='softmax'))

# 编译模型，指定损失函数和优化器
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',  # 多分类交叉熵损失函数
    metrics=['accuracy'])

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 绘制训练损失和验证损失
history_dic = history.history
loss_value = history_dic['loss']
val_loss_values = history_dic['val_loss']
epochs = range(1, len(loss_value) + 1)

plt.plot(epochs, loss_value, 'bo', label='training loss')  # bo为蓝色圆点
plt.plot(epochs, val_loss_values, 'b', label='test loss')  # b为蓝色线
plt.title('training and test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练精度和验证精度
plt.clf()  # 清空之前图像
acc = history_dic['acc']
val_acc = history_dic['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Test acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 上面的模型会出现过拟合，在第9轮之后的训练都朝着过拟合发展了。下面开始一个新的模型，只训练9轮
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
result = model.evaluate(x_test, one_hot_test_labels)
print(result)

# 获取在测试集上的预测结果
print(model.predict(x_test))

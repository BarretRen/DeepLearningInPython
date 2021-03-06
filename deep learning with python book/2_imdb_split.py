# IMDB数据集分类，将电影库的评论分为好和坏两种评论
from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 从数据中选择10000个单词
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)


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
# 将标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 构建神经网络层
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))  # relu将所有负值归零
model.add(layers.Dense(1, activation='sigmoid'))  # sigmoid将任意值压缩到0-1之间

# 编译模型，选择损失函数和优化器
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型, history包含训练过程中所有数据
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=500,
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

# 上面的模型会出现过拟合，在第4轮之后的训练都朝着过拟合发展了。下面开始一个新的模型，只训练4轮
model.fit(x_train, y_train, epochs=4, batch_size=512)
result = model.evaluate(x_test, y_test)
print(result)

# 获取在测试集上的预测结果
print(model.predict(x_test))
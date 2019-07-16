# 利用embedding层在训练网络的同时学习词嵌入
from keras.layers import Embedding, Flatten, Dense
from keras.datasets import imdb
from keras.models import Sequential
from keras import preprocessing
import matplotlib.pyplot as plt

max_features = 10000  # 作为特征的单词个数
maxlen = 20  # 值选取评论的前20个单词

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# 将整数列表转换为二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 在IMDB数据集上使用Embedding层和分类器
model = Sequential()
# 新建embedding层，至少需要两个参数，标记的个数&嵌入的维度
model.add(Embedding(10000, 8, input_length=maxlen))  # 输出形状为(sample, maxlen, 8)
model.add(Flatten())  # 将三位的嵌入张量展平为(sample,maxlen*8)的二位张量
model.add(Dense(1, activation='sigmoid'))  # 添加分类器，分为两类

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

# 训练模型
history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

# 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

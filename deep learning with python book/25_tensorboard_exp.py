# 一维卷积网络处理文本序列,TensorBoard可视化显示
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras import callbacks

max_features = 500  # tensorboard回调是一件占用内存挺高的操作,数据量不要太大
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# 将整数列表转换为二维整数张量
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# 构建一维卷积神经网络
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))  # 添加一维卷积层和池化层
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())  # 添加全局池化层
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

# 添加一个TensorBoard回调函数
callback = [
    callbacks.TensorBoard(
        log_dir='tboard',  # 日志文件位置
        histogram_freq=1,  # 每一轮之后记录激活直方图
        embeddings_freq=1,  # 每一轮之后记录嵌入数据,
        embeddings_data=x_train[:100].astype("float32")  # 要嵌入的数据
    )
]

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callback)

# RNN高级技巧5: 一维CNN和RNN结合
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

# 从文件中获取天气数据集
f = open('./jena_climate_2009_2016.csv')
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

# 解析数据，将数据转换为numpy数组
float_data = np.zeros((len(lines), len(header) - 1))  # 新建一个行列和数据集一致的二维矩阵
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]  # 获取一行数据
    float_data[i, :] = values

# 图示温度和时间的变化关系
# tmp = float_data[:, 1]  # 获取第一列
# plt.plot(range(len(tmp)), tmp)
# plt.show()

# 数据标准化，作用是使每一行的数据符合标准正态分布
# 使用前200000个时间步（行）作为训练数据,所以对前200000进行标准化
mean = float_data[:200000].mean(axis=0)  # 计算每行平均值
float_data -= mean  # 减去平均值
std = float_data[:200000].std(axis=0)  # 计算每行标准差
float_data /= std  # 除以标准差


# 生成时间序列样本的生成器
def generator(
        data,  # 标准化后的原始数组
        lookback,  # 输入数据应该包含多少个时间步（多久的数据）
        delay,  # 返回数据是未来多久的数据
        min_index,  # data数组的索引，与max_index一起缺点抽取哪些时间步
        max_index,
        shuffle=False,  # 是否线打乱样本
        batch_size=128,  # 每个批量的样本数
        step=6):  # 采样频率，结合时间步大小
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback,
                                     max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        sample = np.zeros(
            (len(rows), lookback // step, data.shape[-1]))  # 输入数据的一个批量
        target = np.zeros((len(rows), ))  # 目标温度数组

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            sample[j] = data[indices]
            target[j] = data[rows[j] + delay][1]

        yield sample, target


# 使用上面的生成器生成三个数据集：一个训练，一个验证，一个测试
lookback = 720
step = 3  # step变小，之前是6，因为CNN可以处理更长的序列，所以我们可以没30分钟获取一个数据点，从而分析更多的数据序列
delay = 144
batch_size = 128
train_gen = generator(float_data, lookback, delay, 0, 200000, True, step,
                      batch_size)
val_gen = generator(float_data, lookback, delay, 200001, 300000, False, step,
                    batch_size)
test_gen = generator(float_data, lookback, delay, 300001, None, False, step,
                     batch_size)
# 计算获得数据集的抽取次数
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

# ===========================创建一维CNN和RNN GRU结合的模型
model = Sequential()
# 添加一维CNN和池化层
model.add(
    layers.Conv1D(32,
                  5,
                  activation='relu',
                  input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPool1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(
    train_gen,
    steps_per_epoch=500,
    epochs=20,
    validation_data=val_gen,
    validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

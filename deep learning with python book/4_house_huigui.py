from keras.datasets import boston_housing
import numpy as np
from keras import layers, models

(train_data, train_targets), (test_data,
                              test_targets) = boston_housing.load_data()

# 数据取值范围差异很大，需要借助numpy进行数据标准化
# 即：对于输入数据的每个特征（输入数据矩阵中的列），减去特征平均值，再除以标准差
mean = train_data.mean(axis=0)  # 获取特征平均值
train_data -= mean
std = train_data.std(axis=0)  # 获取标准差
train_data /= std

test_data -= mean  # 这里用训练集得到的mean和std。在深度学习中，不能使用测试集上得到的任何结果
test_data /= std


# 定义模型，因为同一模型需要多测实例化，这里写成一个函数
def build_model():
    model = models.Sequential()
    model.add(
        layers.Dense(64,
                     activation='relu',
                     input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    # 网络的最后一层只有一个单元，没有激活函数，是一个线性层,这是回归的典型设置
    model.add(layers.Dense(1))
    # mse损失函数，即均方误差，是回归问题常用的损失函数
    model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['mae'])  # 监测项为平均绝对误差
    return model


# 使用K折交叉验证来训练模型，由于训练集只有404个样本，所以这里K=4
k = 4
num_val = len(train_data) // k
all_results = []

for i in range(k):
    print('processing fold #', i)
    tmp_data = train_data[i * num_val:(i + 1) * num_val]  # 获取第i个分区的数据
    tmp_targets = train_targets[i * num_val:(i + 1) * num_val]

    # 其他分区数据作为训练数据
    tmp_train_data = np.concatenate(
        [train_data[:i * num_val], train_data[(i + 1) * num_val:]], axis=0)
    tmp_train_targets = np.concatenate(
        [train_targets[:i * num_val], train_targets[(i + 1) * num_val:]],
        axis=0)

    # 开始训练模型
    model = build_model()
    model.fit(tmp_train_data,
              tmp_train_targets,
              epochs=80,
              batch_size=16,
              verbose=0)  # verbose表示静默模式
    val_mse, val_mae = model.evaluate(tmp_data, tmp_targets, verbose=0)
    all_results.append(val_mae)

print(np.mean(all_results))

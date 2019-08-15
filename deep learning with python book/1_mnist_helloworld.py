from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
# 添加两个神经层
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10,
                         activation='softmax'))  # 第二层，输出为10个概率值的数组，对应0-9十个数的概率
network.compile(
    optimizer='rmsprop',  # 设置优化器
    loss='categorical_crossentropy',  # 设置损失函数
    metrics=['accuracy'])  # 要监测的指标

# 对数据集进程预处理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype(
    'float32') / 255  # 变换为一个 float32 数组，其形状为 (60000, 28 * 28)，取值范围为 0~1
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype(
    'float32') / 255  # 变换为一个 float32 数组，其形状为 (10000, 28 * 28)，取值范围为 0~1

# 标签处理
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 得到训练结果，在测试集上验证准确率
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

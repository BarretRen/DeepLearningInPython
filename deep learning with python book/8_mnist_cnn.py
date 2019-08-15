# 使用卷积神经网络模型的mnist分类问题，精度会比密集层高
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = models.Sequential()
# 添加卷积层
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
    28, 28, 1)))  # 卷积神经网络接收形状为 (image_height, image_width, image_channels)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 添加密集层连接到分类器
model.add(layers.Flatten())  # 将图像3d张量变为1d张量
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 多分类，用softmax

model.compile(
    optimizer='rmsprop',  # 设置优化器
    loss='categorical_crossentropy',  # 设置损失函数
    metrics=['accuracy'])  # 要监测的指标

# 对数据集进程预处理
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype(
    'float32') / 255  # 变换为一个 float32 数组，其形状为 (60000, 28, 28, 1)，取值范围为 0~1
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype(
    'float32') / 255  # 变换为一个 float32 数组，其形状为 (10000, 28, 28, 1)，取值范围为 0~1

# 标签处理
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 得到训练结果，在测试集上验证准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# 保存模型
model.save("mnist_cnn.h5")

# 生成式对抗网络示例
import os
import keras
import numpy as np
from keras import layers
from keras.preprocessing import image

# GAN的生成器网络：将来自潜在空间的向量转换为一张候选图像
latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim, ))
# 将输入转换为大小为16x16的128个通道的特征图
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# 上采样为32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# 生成一个大小为32x32的单通道特征图
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)  #实例化模型
generator.summary()

# GAN的判别器网络：接受一张候选图像，判断是生成的图像或真是图像
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)  # 判别器添加一个dropout层，重要技巧
x = layers.Dense(1, activation='sigmoid')(x)  # 分类曾
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
# 优化器使用梯度裁剪，限制梯度值的范围
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008,
                                                   clipvalue=1.0,
                                                   decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')

# 对抗网络：将生成器和判别器连接在一起
# 设置判别器权重为不可训练，不然在训练过程中签字改变，预测结果会始终为真
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim, ))
gan_output = discriminator(generator(gan_input))  # 连接生成器和判别器

gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# 训练GAN
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()  # 加载数据
x_train = x_train[y_train.flatten() == 6]  # 只选择数据集中的青蛙图像
x_train = x_train.reshape((x_train.shape[0], ) +
                          (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir = 'your_dir'  # 生成图像保存路径
start = 0

for step in range(iterations):
    # 在潜在空间随机采样
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 随机采样点通过生成器得到虚假图像
    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start:stop]

    # 将虚假图像和真是图像合在一起
    combined_images = np.concatenate([generated_images, real_images])
    # 合并标签区分真实和虚假图像
    labels = np.concatenate(
        [np.ones((batch_size, 1)),
         np.zeros((batch_size, 1))])
    # 想标签添加噪声，重要技巧
    labels += 0.05 * np.random.random(labels.shape)
    # 合并图像用于训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # 在潜在空间再随机采样一次
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1))  # 标签设置为全真
    # 通过GAN训练生成器
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:  # 每100步保存并绘图
        gan.save_weights('gan.h5')  # 保存权重

        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        # 保存虚假图像
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        # 保存真实图像
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))

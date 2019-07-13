# 可视化中间输出
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import models
import numpy as np
from keras.preprocessing import image

model = load_model('cats_dogs.h5')  # 加载之前保存的卷积模型

# 预处理单张图片
img = image.load_img('./cats.jpg', target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

layer_outputs = [layer.output for layer in model.layers[:8]]  # 提取前8层的输出
# 创建一个Model用于可视化输出，输入和输出都来自卷积模型
view_model = models.Model(inputs=model.input, outputs=layer_outputs)
# 运行可视化Model
views = view_model.predict(img_tensor)  # 这里使用mnist测试集的第一个图像

# 可视化每个中间激活的通道
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)  # 保存每一层的名字

images_per_row = 16  # 一行显示16个图像

for layer_name, layer_view in zip(layer_names, views):
    n_features = layer_view.shape[-1]  # 特征图中的特征个数

    size = layer_view.shape[1]  # 特征图的形状为 (1, size, size, n_features)

    n_rows = n_features // images_per_row  # 行数
    display_grid = np.zeros((size * n_rows, images_per_row * size))

    for row in range(n_rows):
        for col in range(images_per_row):
            channel_image = layer_view[0, :, :, row * images_per_row + col]
            # 对特征进程处理，使其看起来更美观
            # channel_image -= channel_image.mean()
            # channel_image /= channel_image.std()
            # channel_image *= 64
            # channel_image += 128
            # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # 填充网格
            display_grid[row * size:(row + 1) * size, col * size:(col + 1) *
                         size] = channel_image

    scale = 1. / size  # 缩小比例
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
# 可视化每层的过滤器
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model


# 将张量转换为有效图像的实用函数：输入格式为(,,,)
def deprocess_image(x):
    # 对张量做标准化，使其均值为0，标准差为0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # 将x裁切到[0,1]区间
    x += 0.5
    x = np.clip(x, 0, 1)

    # 将x转换为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 构建一个损失函数，将该层第n个过滤器的激活最大化
def generate_pattern(model, layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # 计算这个损失相对于输入图像的梯度
    grads = K.gradients(loss, model.input)[0]
    # 对梯度进行L2标准化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # 返回给定输入图像的损失和梯度
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128

    # 进行40次梯度上升
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


model = load_model('cats_dogs.h5')  # 加载之前保存的卷积模型
layer_name = 'conv2d_2'
filter_index = 0

# 生成第4层卷积的64个过滤器,每个网格是一个 64 像素×64 像素的过滤器模式
size = 64
margin = 5
# 保存结果
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(model,
                                      layer_name,
                                      i + (j * 8),
                                      size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start:horizontal_end, vertical_start:
                vertical_end, :] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()

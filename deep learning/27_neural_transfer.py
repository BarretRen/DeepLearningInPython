# 基于VGG19卷积网络的诗经风格迁移
import time
import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import img_to_array, load_img
import cv2
from scipy.optimize import fmin_l_bfgs_b

target_path = './base.png'  # 要变换的图像的路径
style_path = './style.png'  # 风格图像的路径

width, height = load_img(target_path).size
img_height = 400  # 限定宽高
img_width = int(width * img_height / height)


# 辅助函数
def preprocess_img(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_img(x):
    # vgg19.preprocess_input的逆操作
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 将BGR格式转换为RGB格式图像，也是vgg19.preprocess_input逆操作的一部分
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 加载与训练的VGG19网络，应用于三张图像
target_image = K.constant(preprocess_img(target_path))
style_image = K.constant(preprocess_img(style_path))
combind_image = K.placeholder((1, img_height, img_width, 3))
# 将三张图像合并为一个批量
input_tensor = K.concatenate([target_image, style_image, combind_image],
                             axis=0)
# 作为VGG19网络模型的输入，进行训练
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded')


# 定义内容损失，保证目标图像和生成图像在 VGG19 卷积神经网络的顶层具有相似的结果
def content_loss(base, combind):
    return K.sum(K.square(combind - base))


# 定义风格损失
# 使用一个辅助函数来计算输入矩阵的格拉姆矩阵
def gram_matrix(x):
    feature = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(feature, K.transpose(feature))
    return gram


def style_loss(style, combind):
    S = gram_matrix(style)
    C = gram_matrix(combind)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels**2) * (size**2))


# 计算总变差损失，对生成的组合图像的像素进行操作。它促使生成图像具有空间连续性，从而避免结果过度像素化
def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 获取模型中各层的字典，层名称和层输出
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'  # 内容损失，只用VGG19的顶层
# 风格损失，需要用VGG19的所有层
style_layers = [
    'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1',
    'block5_conv1'
]
# 计算loss所用的权重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# 定义最终损失，并添加内容损失
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)

# 添加每一层的风格损失
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

# 添加总变差损失
loss += total_variation_weight * total_variation_loss(combind_image)

# 设置梯度下降过程
grads = K.gradients(loss, combind_image)[0]  # 获取损失相对于生成图像的梯度
fetch_loss_and_grads = K.function([combind_image],
                                  [loss, grads])  # 获取当前损失之和当前梯度值的函数


# 类，将 fetch_loss_and_grads 包装起来，让你可以利用两个单独的方法调用来获取损失和梯度，这是我们要使用的 SciPy 优化器所要求的
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()  # 实例化类

# 神经风格迁移循环
# 使用L_BFGS算法运行梯度上升，在算法每一次迭代时白痴当前的生成图像
result_prefix = 'my_result'
iterations = 20

x = preprocess_img(target_path)
x = x.flatten()  # 将图像展平，fmin_l_bfgs_b 只能处理展平的向量
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    # 运行L_BFGS算法，将神经风格损失最小化，此处传入自定义类的loss和grads函数
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    # 保存当前图像
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_img(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    cv2.imwrite(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

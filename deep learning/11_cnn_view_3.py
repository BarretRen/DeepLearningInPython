# 可视化类激活热力图
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras import backend as K
import cv2

model = load_model('cats_dogs.h5')  # 加载之前保存的卷积模型

# 预处理单张图片
img = image.load_img('./cats.jpg', target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# 得到模型结果
preds = model.predict(img_tensor)

# 应用Grad-CAM算法
cat_output = model.output[:, 0]  # 得到输出中猫类别，0通过np.argmax(preds[0])获取
last_conv_layer = model.get_layer('conv2d_4')  # 层名
# 计算猫类别对于第4层输出特征图的梯度
grads = K.gradients(cat_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
# 将特征图数组的每个通道乘以通道对猫类别的重要程度
for i in range(128):  # 128为图像大小
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 得到类激活热力图
heatmap = np.mean(conv_layer_output_value, axis=-1)
# 显示热力图
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
# 将热力图与原图叠加
img = cv2.imread('./cats.jpg')  # 加载原图
# 将热力图大小调整为与原图相同
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# 热力图转换为RGB格式
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./cat_cam.jpg', superimposed_img)

# 使用循环神经网络LSTM层生成序列文本
import random
import sys
import keras
import numpy as np
from keras import layers

# 下载解析初始文本，这里用尼采的作品数据
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().loswe()  # 读取所有文本，转换为小写

# 将字符向量化，one-hot编码
max_len = 60  # 提取60个字符组成的序列
step = 3  # 每三个字符采样一个新序列
sentences = []  # 保存提取的序列
next_chars = []  # 保存下一目标

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i:i + max_len])
    next_chars.append(text[i + max_len])

print('number of sequences:', len(sentences))

chars = sorted(list(set(text)))
# 字典，将唯一字符映射为索引位置
char_indices = dict((char, chars.index(char)) for char in chars)

# 对字符one-hot编码
x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 构建网络
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(max_len, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# 函数：给定模型预测结果，采样下一个字符
def selectNextChar(perds, temperature=1.0):
    preds = np.asarray(perds).astype('float64')
    # 根据softmax温度，对分布重新加权
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)  # 指数运算
    preds = exp_preds / np.sum(exp_preds)
    # 加权后，对下一字符进行随机采样
    index = np.random.multinomial(1, preds, 1)
    return np.argmax(index)  # 返回下一字符index


# 文本生成循环
for epoch in range(1, 60):  # 训练60轮
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)  # 模型在数据上训练一次
    # 随机选择一个文本种子
    start_index = random.randint(0, len(text) - max_len - 1)
    generated_text = text[start_index:start_index + max_len]
    print('--- Generating with seed: "' + generated_text + '"')
    for temperature in [0.2, 0.5, 1.0, 1.2]:  # 尝试不同的采样温度
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        for i in range(400):  # 从种子文本开始，生成400个字符
            sampled = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.  # one-hot编码

            preds = model.predict(sampled, verbose=0)[0]  # 根据模型训练的权重，获取验证结果
            next_index = selectNextChar(preds, temperature)  # 放入采样函数，获取下一字符
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)

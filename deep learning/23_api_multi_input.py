# 用Keras 函数API实现多输入模型，Sequential模型是单输入的，无法实现
from keras.models import Model
from keras import layers
from keras import Input
from keras import utils
import numpy as np

# 定义序列的大小
text_size = 10000
question_size = 10000
answer_size = 500
# 模型的文本输入，一个长度可变的证书序列
text_input = Input(shape=(None, ), dtype='int32', name='text')
# 创建词嵌入，将文本输入嵌入长度64的向量
embedded_text = layers.Embedding(text_size, 64)(text_input)
# 对文本输入向量添加循环层LSTM
encoded_text = layers.LSTM(32)(embedded_text)

# 对问题输入执行相同的操作
question_input = Input(shape=(None, ), dtype='int32', name='question')
embedded_question = layers.Embedding(question_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# 将编码后的两个输入连接在一起
conbinded = layers.concatenate([encoded_text, encoded_question], axis=-1)
# 添加分类器
answer = layers.Dense(answer_size, activation='softmax')(conbinded)

# 实例化模型，指定输入和输出
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 生成虚拟数据，进行训练
num_samples = 1000
max_length = 100

text = np.random.randint(1, text_size, size=(num_samples, max_length))
question = np.random.randint(1, question_size, size=(num_samples, max_length))
answers = np.random.randint(answer_size, size=(num_samples))
answers = utils.to_categorical(answers, answer_size)  # answers转化为one-hot编码

model.fit([text, question], answers, epochs=10, batch_size=128)  # 参数要与模型实例化时一致

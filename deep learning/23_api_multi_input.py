# 用Keras 函数API实现多输入模型，Sequential模型是单输入的，无法实现
from keras.models import Model
from keras import layers
from keras import Input

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

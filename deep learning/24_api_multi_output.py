# 用Keras 函数API实现多输出模型，Sequential模型是单输入的，无法实现
from keras.models import Model
from keras import layers
from keras import Input
from keras import utils
import numpy as np

vocabulary_size = 50000
num_income_groups = 10

# 模型的输入
post_input = Input(shape=(None, ), dtype='int32', name='posts')
# 创建词嵌入，将文本输入嵌入长度64的向量
embedded_posts = layers.Embedding(256, vocabulary_size)(post_input)

# 添加以为卷积和池化
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
# 添加分类器
x = layers.Dense(128, activation='relu')(x)

# 输出层
age_output = layers.Dense(1, name='age')(x)
income_output = layers.Dense(num_income_groups,
                             activation='softmax',
                             name='income')
gender_output = layers.Dense(1, actiation='sigmoid', name='gender')

# 实例化模型，指定输入和输出
model = Model(post_input, [age_output, income_output, gender_output])
# 由于有多个输出，所以需要进行多重损失函数的设置
model.compile(
    optimizer='rmsprop',
    loss={  # 维每个输出指定损失函数
        'age': 'mse',
        'income': 'categorical_crossentropy',
        'gender': 'binary_crossentropy'
    },
    loss_weights={  # 为每个输出添加权重，防止对单个损失进行过度优化
        'age': 0.25,
        'income': 1.,
        'gender': 10.
    })

# 进行训练
model.fit(posts, [age_targets, income_targets, gender_targets],
          epochs=10,
          batch_size=64)  # 参数要与模型实例化时一致

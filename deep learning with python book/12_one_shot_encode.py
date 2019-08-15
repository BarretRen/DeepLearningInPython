# 演示文本单词和字符的one-hot编码
import numpy as np
import string
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# ====================对单词进行编码
token_index = {}  # 构建数据中所有标记的索引
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1  # 为每个唯一单词指定唯一索引

max_length = 10  # 对前十个单词进行one-hot编码
# 保存编码在results里
results = np.zeros(shape=(len(samples), max_length,
                          max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print(results)

# =======================对字符进行one-hot编码
characters = string.printable  # 获取所有可打印的ASCII字符
token_index = dict(zip(range(1, len(characters) + 1), characters))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

print(results)

# =============================keras实现单词one-hot编码
# 创建一个分词器，设置值考虑前1000个单词
tokenizer = Tokenizer(num_words=1000)
# 构建单词索引
tokenizer.fit_on_texts(samples)
# 将字符串转换为整数索引列表
sequences = tokenizer.texts_to_sequences(samples)
# 得到one-hot二进制编码
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print(one_hot_results)
# 得到单词索引
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# ==========================one-hot哈希编码
dimensionality = 1000  # 哈希长度为1000，如果单词个数接近1000，会出现哈希冲突，暴露缺点
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

print(results)

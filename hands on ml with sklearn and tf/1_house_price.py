# 第一个示例，预测加州某个地区的房价
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 随机分类方法
from sklearn.model_selection import StratifiedShuffleSplit  # 分层分类方法


# 函数：加载csv文件，返回dataframe对象
def load_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_data("hands on ml with sklearn and tf\\housing")
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())  # 查看所有类别，因为csv中这一项是对象，不是数值
print(housing.describe())
# 显示每列属性的柱状图
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# 随机采样：使用sk-learn中的方法生成测试集和验证集
# test_size为两个数据集的比例，random_state为随机种子
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# 根据median_income添加新列，分成5类，median_income大于5的当作5处理
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
print(housing["income_cat"].value_counts() / len(housing))

# 分层采样：根据收入等级，用sk-learn的方法生成测试集和验证集
# 分层采样测试集的收入分类比例与总数据集几乎相同， 而随机采样数据集偏差严重
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 删除新加的income_cat列
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

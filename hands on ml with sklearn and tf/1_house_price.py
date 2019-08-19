# 第一个示例，预测加州某个地区的房价
import os
import pandas as pd
import matplotlib.pyplot as plt


# 函数：加载csv文件，返回dataframe对象
def load_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_data("./housing")
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())  # 查看所有类别，因为csv中这一项是对象，不是数值
print(housing.describe())
# 显示每列属性的柱状图
housing.hist(bins=50, figsize=(20, 15))
plt.show()

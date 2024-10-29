import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

'''
github：https://github.com/boating-in-autumn-rain?tab=repositories
网址：www.slothai.cn
微信公众号：秋雨行舟
B站：秋雨行舟
抖音：秋雨行舟
咨询微信：slothalone
'''


# 数据预处理代码
def prepro(file_path, spilt_rate):
    # 读取xls文件
    df = pd.read_csv(file_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 分割特征和标签
    X = df.iloc[:, 3:-1]  # 特征数据
    y = df.iloc[:, -1]   # 标签数据

    # 划分数据集
    # 先将数据分为训练集和剩余部分（测试集+验证集）
    x_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-spilt_rate, stratify=y, random_state=42)

    # 再将剩余部分划分为测试集和验证集
    x_valid, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 初始化归一化器，将范围设为 0-255
    scaler = MinMaxScaler(feature_range=(0, 255))

    # 只用训练集拟合归一化器，然后分别转换训练集和测试集
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_valid = scaler.transform(x_valid)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)

    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_valid, y_val, x_test, y_test


# github：https://github.com/boating-in-autumn-rain?tab=repositories
# 微信公众号：秋雨行舟
# B站：秋雨行舟
# 抖音：秋雨行舟
# 咨询微信：slothalone


from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

# 数据预处理代码
def prepro(d_path, length, number, normal, rate):

    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    # 取DE的数据
    def capture(original_path):
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    # 训练数据和测试数据集划分
    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            samp_train = int(number * (1 - slice_rate))
            Train_sample = []
            Test_Sample = []

            for j in range(samp_train):
                sample = slice_data[j*150: j*150 + length]
                Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                sample = slice_data[samp_train*150 + length + h*150: samp_train*150 + length + h*150 + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    # 数据标准化，可选
    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        data_all = np.vstack((Train_X, Test_X))
        scalar = preprocessing.StandardScaler().fit(data_all)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X


    # 将测试集划分为测试集和验证集，形成训练集、验证集、测试集三部分
    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)

        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]

        return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)

    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)

    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)

    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)

    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)

    #  转为array类型
    Train_X = np.asarray(Train_X)
    Test_X = np.asarray(Test_X)

    # 将测试集切分为验证集和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y



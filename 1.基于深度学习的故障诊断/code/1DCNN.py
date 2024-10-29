import matplotlib.pyplot as plt
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow as tf
import preprocessing
from sklearn.metrics import classification_report
import warnings
warnings .filterwarnings("ignore")
# 设置字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''
github：https://github.com/boating-in-autumn-rain?tab=repositories
网址：www.slothai.cn
微信公众号：秋雨行舟
B站：秋雨行舟
抖音：秋雨行舟
咨询微信：slothalone
'''

# 绘制acc和loss曲线
def acc_loss_line(history):
    print("绘制准确率和损失值曲线")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 绘制accuracy曲线
    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('训练集和验证集准确率曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.legend(["训练集准确率", "验证集准确率"])

    plt.figure()

    # 绘制loss曲线
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('训练集和验证集损失值曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.legend(["训练集损失值", "验证集损失值"])

    plt.show()


# 对输入到模型中的数据进一步处理
def data_pre(path, length, number, normal, rate, stride):
    # 得到训练集、验证集、测试集
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocessing.prepro(d_path=path, length=length, number=number,
                                                                              normal=normal, rate=rate, stride=stride)
    # 转为数组array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # 标签转为int
    y_train = [int(i) for i in y_train]
    y_valid = [int(i) for i in y_valid]
    y_test = [int(i) for i in y_test]

    # 打乱顺序
    index = [i for i in range(len(x_train))]
    random.seed(1)
    random.shuffle(index)
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]

    index1 = [i for i in range(len(x_valid))]
    random.shuffle(index1)
    x_valid = np.array(x_valid)[index1]
    y_valid = np.array(y_valid)[index1]

    index2 = [i for i in range(len(x_test))]
    random.shuffle(index2)
    x_test = np.array(x_test)[index2]
    y_test = np.array(y_test)[index2]

    x_train = tf.reshape(x_train, (len(x_train), length, 1))
    x_valid = tf.reshape(x_valid, (len(x_valid), length, 1))
    x_test = tf.reshape(x_test, (len(x_test), length, 1))

    return x_train, y_train, x_valid, y_valid, x_test, y_test

# 模型定义
def mymodel(x_train):
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    h1 = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(h1)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Conv1D(filters=8, kernel_size=4, strides=1, padding='same', activation='relu')(h1)

    h1 = layers.Dropout(0.6)(h1)
    h1 = layers.Flatten()(h1)
    h1 = layers.Dense(64, activation='relu')(h1)
    h1 = layers.Dense(10, activation='softmax')(h1)

    deep_model = keras.Model(inputs, h1, name="cnn")
    return deep_model

def model_train(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, epochs):
    model = mymodel(x_train)

    # 打印模型参数
    model.summary()

    # 编译模型
    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 模型训练
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_valid, y_valid))

    # 评估模型
    scores = model.evaluate(x_test, y_test, verbose=1)

    print("=========模型训练结束==========")
    print("测试集结果： ", '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    y_predict = model.predict(x_test)
    y_pred_int = np.argmax(y_predict, axis=1)

    print("混淆矩阵输出结果：")
    print(classification_report(y_test, y_pred_int, digits=4))

    return history, model


# main函数
if __name__ == '__main__':
    length = 784  # 样本长度
    number = 600  # 每类样本的数量
    normal = True  # 是否标准化
    rate = [0.7, 0.15, 0.15]  # 训练集、测试集、验证集的划分比例
    path = r'../data/0HP'  # 数据集路径
    stride = 150  # 滑动步长
    batch_size = 128 #  训练批次
    epochs = 100  # 训练轮次

    # 1.数据集预处理
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_pre(path, length, number, normal, rate, stride)

    # 2.一维卷积神经网络模型训练
    history, model = model_train(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, epochs)

    # 3.准确率与损失值曲线展示
    acc_loss_line(history)

# github：https://github.com/boating-in-autumn-rain?tab=repositories
# 微信公众号：秋雨行舟
# B站：秋雨行舟
# 抖音：秋雨行舟
# 咨询微信：slothalone

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import preprocessing
from sklearn.metrics import classification_report
import warnings
warnings .filterwarnings("ignore")

#如果是GPU，需要去掉注释，如果是CPU，则注释
# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)

# 保存最佳模型自定义类
class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss

# t-sne初始可视化函数
def start_tsne(x_train):
    print("正在进行初始输入数据的可视化...")
    x_train1 = tf.reshape(x_train, (len(x_train), 784))
    X_tsne = TSNE().fit_transform(x_train1)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.show()

# t-sne结束可视化函数
def end_tsne(x_test):
    model.load_weights(filepath='../model/1DCNN.h5')
    hidden_features = model.predict(x_test)
    X_tsne = TSNE().fit_transform(hidden_features)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test)
    plt.colorbar()
    plt.show()


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


# 绘制acc和loss曲线
def acc_line():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 绘制accuracy曲线
    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])

    plt.figure()

    # 绘制loss曲线
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.show()


# 绘制混淆矩阵
def confusion(x_test, y_test):
    y_pred_gailv = model.predict(x_test, verbose=1)
    y_pred_int = np.argmax(y_pred_gailv, axis=1)
    con_mat = confusion_matrix(y_test.astype(str), y_pred_int.astype(str))
    print(con_mat)
    classes = list(set(y_train))
    classes.sort()
    plt.imshow(con_mat, cmap=plt.cm.Blues)
    indices = range(len(con_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('guess')
    plt.ylabel('true')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.show()


# 对输入到模型中的数据进一步处理
def data_pre():
    length = 784  # 样本长度
    number = 500  # 每类样本的数量
    normal = True  # 是否标准化
    rate = [0.5, 0.25, 0.25]  # 训练集、测试集、验证集的划分比例
    path = r'../data/0HP'  # 数据集路径

    # 得到训练集、验证集、测试集
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocessing.prepro(d_path=path, length=length, number=number,
                                                                              normal=normal, rate=rate)
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

    x_train = tf.reshape(x_train, (len(x_train), 784, 1))
    x_valid = tf.reshape(x_valid, (len(x_valid), 784, 1))
    x_test = tf.reshape(x_test, (len(x_test), 784, 1))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# main函数
if __name__ == '__main__':

    # 获取数据
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_pre()

    print("x_train.shape: ", x_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("x_valid.shape: ", x_valid.shape)
    print("y_valid.shape: ", y_valid.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_test.shape: ", y_test.shape)

    # t-sne初始可视化
    start_tsne(x_train)

    # 获取定义模型
    model = mymodel(x_train)

    # 打印模型参数
    model.summary()

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # 模型训练
    history = model.fit(x_train, y_train,
                        batch_size=128, epochs=200, verbose=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[CustomModelCheckpoint(
      model, r'../model/1DCNN.h5')])

    # 加载模型
    model.load_weights(filepath='../model/1DCNN.h5')

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # 评估模型
    scores = model.evaluate(x_test, y_test, verbose=1)

    print("=========模型训练结束==========")
    print("测试集结果： ", '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    y_predict = model.predict(x_test)
    y_pred_int = np.argmax(y_predict, axis=1)

    print("混淆矩阵输出结果：")
    print(classification_report(y_test, y_pred_int, digits=4))

    # 绘制acc和loss曲线
    print("绘制acc和loss曲线")
    acc_line()

    # 训练结束的t-sne降维可视化
    print("训练结束的t-sne降维可视化")
    end_tsne(x_test)

    # 绘制混淆矩阵
    print("绘制混淆矩阵")
    confusion(x_test, y_test)
import time
import pandas as pd
from keras import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import warnings
import preprocessing

warnings.filterwarnings("ignore")
# 设置字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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


def start_tsne(x_data, y_data, length, csv_filename, image_filename):
    # t-SNE初始可视化
    print("正在进行初始输入数据的可视化...")
    x_data_reshaped = tf.reshape(x_data, (len(x_data), length))
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(x_data_reshaped)

    tsne_df = pd.DataFrame(X_tsne, columns=["维度1", "维度2"])
    tsne_df['标签'] = y_data
    tsne_df.to_csv(csv_filename, index=False)
    print(f"t-SNE 结果已保存到 {csv_filename}")

    plt.figure(figsize=(10, 10))
    cmap = plt.cm.get_cmap('tab20')
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_data, cmap=cmap, alpha=0.7, s=60, edgecolors='k')
    unique_labels = np.unique(y_data)

    for label in unique_labels:
        plt.scatter([], [], c=scatter.cmap(scatter.norm(label)), label=f"{label}", s=100, edgecolors='black')

    plt.title("T-SNE 可视化", fontsize=16)
    plt.xlabel("维度 1", fontsize=16)
    plt.ylabel("维度 2", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="标签", fontsize=16, loc="best", markerscale=1, frameon=True)
    plt.savefig(image_filename, dpi=600, bbox_inches='tight')
    print(f"图片已保存到 {image_filename}")
    plt.show()


def end_tsne(x_data, y_data, csv_filename, model_file, end_tsne_image_filename):
    # t-SNE结束可视化
    print("训练结束的t-sne降维可视化")
    model.load_weights(filepath=model_file)
    hidden_features = model.predict(x_data)

    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(hidden_features)
    tsne_df = pd.DataFrame(X_tsne, columns=["维度1", "维度2"])
    tsne_df['标签'] = y_data
    tsne_df.to_csv(csv_filename, index=False)
    print(f"t-SNE 结果已保存到 {csv_filename}")

    plt.figure(figsize=(10, 10))
    cmap = plt.cm.get_cmap('tab20')
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_data, cmap=cmap, alpha=0.7, s=60, edgecolors='k')
    unique_labels = np.unique(y_data)

    for label in unique_labels:
        plt.scatter([], [], c=scatter.cmap(scatter.norm(label)), label=f"{label}", s=100, edgecolors='black')

    plt.title("T-SNE 可视化", fontsize=16)
    plt.xlabel("维度 1", fontsize=16)
    plt.ylabel("维度 2", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="标签", loc="best", markerscale=1, frameon=True)
    plt.savefig(end_tsne_image_filename, dpi=600, bbox_inches='tight')
    print(f"图片已保存到 {end_tsne_image_filename}")
    plt.show()


def acc_loss_line(history, acc_loss_filename, acc_image_filename, loss_image_filename):
    print("绘制准确率和损失值曲线")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 保存数据到CSV文件
    data = {
        'epoch': list(epochs),
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss
    }
    df = pd.DataFrame(data)
    df.to_csv(acc_loss_filename, index=False)

    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('训练集和验证集准确率曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.legend(["训练集准确率", "验证集准确率"])
    plt.savefig(acc_image_filename, dpi=600, bbox_inches='tight')
    print(f"图片已保存到 {acc_image_filename}")

    plt.figure()
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('训练集和验证集损失值曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.legend(["训练集损失值", "验证集损失值"])
    plt.savefig(loss_image_filename, dpi=600, bbox_inches='tight')
    print(f"图片已保存到 {loss_image_filename}")
    plt.show()


def confusion(x_test, y_test, model, confusion_filename, confusion_image_filename):
    print("绘制混淆矩阵")
    y_predict = model.predict(x_test)
    y_pred_int = np.argmax(y_predict, axis=1)

    print("混淆矩阵输出结果：")
    print(classification_report(y_test, y_pred_int, digits=4))

    con_mat = confusion_matrix(y_test.astype(str), y_pred_int.astype(str))
    con_mat_percent = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis] * 100
    con_mat_percent_df = pd.DataFrame(con_mat_percent,
                                      index=[f'真实_{cls}' for cls in np.unique(y_test)],
                                      columns=[f'预测_{cls}' for cls in np.unique(y_pred_int)])

    con_mat_percent_df.to_csv(confusion_filename, float_format='%.1f')
    print(f"混淆矩阵已保存到 {confusion_filename}")

    classes = list(set(y_test))
    classes.sort()

    plt.figure(figsize=(10, 7))
    plt.imshow(con_mat_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵', fontsize=16)
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    threshold = con_mat_percent.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{con_mat_percent[i, j]:.1f}%",
                     ha='center', va='center',
                     color='white' if con_mat_percent[i, j] > threshold else 'black')

    plt.ylabel('真实标签', fontsize=16)
    plt.xlabel('预测标签', fontsize=16)
    plt.tight_layout()
    plt.savefig(confusion_image_filename, dpi=600, bbox_inches='tight')
    print(f"图片已保存到 {confusion_image_filename}")
    plt.show()


def data_pre(file_path, spilt_rate, max_words, max_len):
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocessing.prepro(file_path, spilt_rate, max_words, max_len)

    y_train = [int(i) for i in y_train]
    y_valid = [int(i) for i in y_valid]
    y_test = [int(i) for i in y_test]

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

    x_train = tf.reshape(x_train, (len(x_train), max_len))
    x_valid = tf.reshape(x_valid, (len(x_valid), max_len))
    x_test = tf.reshape(x_test, (len(x_test), max_len))

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def textcnn_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # 嵌入层
    embedding = layers.Embedding(input_dim=20000, output_dim=128)(inputs)  # 词汇表大小为20000

    # 卷积层与池化层
    conv_1 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(embedding)
    # conv_1 = layers.BatchNormalization()(conv_1)  # 加入BatchNormalization
    pool_1 = layers.MaxPooling1D(pool_size=2)(conv_1)
    pool_1 = layers.Dropout(0.5)(pool_1)  # 在池化层之后加Dropout

    conv_2 = layers.Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')(embedding)
    # conv_2 = layers.BatchNormalization()(conv_2)  # 加入BatchNormalization
    pool_2 = layers.MaxPooling1D(pool_size=2)(conv_2)
    pool_2 = layers.Dropout(0.5)(pool_2)  # 在池化层之后加Dropout

    conv_3 = layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(embedding)
    # conv_3 = layers.BatchNormalization()(conv_3)  # 加入BatchNormalization
    pool_3 = layers.MaxPooling1D(pool_size=2)(conv_3)
    pool_3 = layers.Dropout(0.5)(pool_3)  # 在池化层之后加Dropout

    # 合并多个卷积池化结果
    merge = layers.concatenate([pool_1, pool_2, pool_3], axis=1)
    flatten = layers.Flatten()(merge)

    # Dropout层
    flatten = layers.Dropout(0.6)(flatten)

    # 全连接层
    dense = layers.Dense(128, activation='relu')(flatten)
    dense = layers.Dropout(0.5)(dense)  # 在全连接层后加Dropout

    # 输出层
    output = layers.Dense(2, activation='softmax')(dense)

    # 构建模型
    model = models.Model(inputs=inputs, outputs=output)

    return model


def model_train(x_train, y_train, x_valid, y_valid, x_test, y_test, save_model_filename, batch_size, epochs):
    model = textcnn_model((x_train.shape[1],))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[CustomModelCheckpoint(model, save_model_filename)])

    return history, model


def model_test(model, save_model_filename):
    model.load_weights(filepath=save_model_filename)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("测试集结果：", '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    file_path = '../../data/IMDB/IMDB_Dataset.csv'
    max_words = 20000
    max_len = 200
    spilt_rate = 0.7  # 训练集、测试集、验证集的划分比例

    start_tsne_csv_filename = "../../save_csv/IMDB/TextCNN/start_tsne_data.csv"
    end_tsne_csv_filename = "../../save_csv/IMDB/TextCNN/end_tsne_data.csv"
    save_model_filename = "../../save_model/IMDB/TextCNN.h5"
    batch_size = 256  # 训练批次
    epochs = 10  # 训练轮次
    acc_loss_filename = '../../save_csv/IMDB/TextCNN/training_history.csv'
    confusion_filename = '../../save_csv/IMDB/TextCNN/confusion.csv'

    start_tsne_image_filename = '../../save_picture/IMDB/TextCNN/start_tsne.png'
    end_tsne_image_filename = '../../save_picture/IMDB/TextCNN/end_tsne.png'
    acc_image_filename = '../../save_picture/IMDB/TextCNN/acc.png'
    loss_image_filename = '../../save_picture/IMDB/TextCNN/loss.png'
    confusion_image_filename = '../../save_picture/IMDB/TextCNN/confusion.png'

    # 1.数据集预处理
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_pre(file_path, spilt_rate, max_words, max_len)

    # 2.训练开始的t-sne降维可视化
    start_tsne(x_train, y_train, max_len, start_tsne_csv_filename, start_tsne_image_filename)

    # 3.TextCNN模型训练
    history, model = model_train(x_train, y_train, x_valid, y_valid, x_test, y_test, save_model_filename, batch_size, epochs)

    # 4.模型测试
    model_test(model, save_model_filename)

    # 5.准确率与损失值曲线展示
    acc_loss_line(history, acc_loss_filename, acc_image_filename, loss_image_filename)

    # 6.训练结束的t-sne降维可视化
    end_tsne(x_train, y_train, end_tsne_csv_filename, save_model_filename, end_tsne_image_filename)

    # 7.混淆矩阵展示
    confusion(x_test, y_test, model, confusion_filename, confusion_image_filename)

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 定义一个函数来清洗文本数据（去除HTML标签等）
def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<br />', ' ', text)  # 替换 <br /> 标签为空格
    text = re.sub(r'<.*?>', '', text)  # 去除所有HTML标签
    # 去除其他不必要的字符（可根据需要调整）
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()  # 转为小写
    return text


# 数据预处理代码
def prepro(file_path, spilt_rate, max_words, max_len):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 分割特征和标签
    X = df.iloc[:, 0]  # 特征数据（评论文本）
    y = df.iloc[:, -1]  # 标签数据（情感标签）

    # 将情感标签转换为 0 和 1：negative -> 0, positive -> 1
    y = y.map({'negative': 0, 'positive': 1})

    # 清洗评论文本数据
    X = X.apply(clean_text)

    # 划分数据集：先将数据分为训练集和剩余部分（测试集+验证集）
    x_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - spilt_rate, stratify=y)

    # 再将剩余部分划分为测试集和验证集
    x_valid, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    # 创建Tokenizer来对文本进行分词
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(x_train)

    # 将文本转换为数字序列
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_valid_seq = tokenizer.texts_to_sequences(x_valid)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    # 对文本序列进行填充，使得所有文本序列的长度相同
    x_train = pad_sequences(x_train_seq, maxlen=max_len)
    x_valid = pad_sequences(x_valid_seq, maxlen=max_len)
    x_test = pad_sequences(x_test_seq, maxlen=max_len)

    # 输出数据集的大小以确认划分
    print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
    print(f"x_valid.shape: {x_valid.shape}, y_val.shape: {y_val.shape}")
    print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")

    return x_train, y_train, x_valid, y_val, x_test, y_test


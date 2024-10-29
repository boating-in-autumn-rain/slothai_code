github：https://github.com/boating-in-autumn-rain?tab=repositories

微信公众号：秋雨行舟

B站：秋雨行舟

咨询微信：slothalone

### 第一章：安装以及环境配置

#### 1.Pycharm安装

```
官网: https://www.jetbrains.com/pycharm/
```

#### 2.Anaconda安装与配置

```
官网：https://www.anaconda.com/download#downloads

配置环境变量（按照自己的安装路径进行配置即可）
D:\software\anaconda\install
D:\software\anaconda\install\Scripts
D:\software\anaconda\install\Library\bin
D:\software\anaconda\install\Library\mingw-w64\bin
```

#### 3.Tensorflow安装

```
官网：https://tensorflow.google.cn/?hl=zh-cn

创建深度学习环境并命名为tf2，指定python版本为3.11:
conda create --name tf2 python=3.11

Anaconda激活tf2环境:
conda activate tf2

Anaconda退出激活环境:
conda deactivate


安装Tensorflow并指定版本为2.15，使用豆瓣源进行加速
pip install tensorflow==2.15.0 -i https://pypi.douban.com/simple/

安装scikit-learn，使用豆瓣源进行加速
pip install scikit-learn -i https://pypi.douban.com/simple/
```



### 第二章：代码讲解与运行调试

#### 0.数据集说明

```
凯斯西楚大学轴承数据集官网：
https://engineering.case.edu/bearingdatacenter/download-data-file

数据集说明(翻译于官网数据集说明)：
收集了正常轴承、单点驱动端和风扇端缺陷的数据。驱动端轴承实验以12000个样本/秒和48000个样本/秒的速度采集数据。所有风机端轴承数据以12000个样本/秒的速度采集。

数据文件为Matlab格式。每个文件包含风扇和驱动端振动数据以及电机转速。对于所有文件，变量名中的以下项表示:

数据集包括：
(1)正常基线数据
(2)12k驱动端轴承故障数据
(3)48k传动端轴承故障数据
(4)风机端轴承故障数据

实验数据集仅使用12k驱动端轴承故障数据，其余数据集预处理与实验数据预处理一致。
```



#### 1.基于卷积神经网络的故障诊断（1DCNN、2DCNN）

```
文件夹说明：
data：数据集存放路径
save_picture/1DCNN:存放1DCNN的图片
save_picture/2DCNN:存放2DCNN的图片

model/1DCNN.h5: 保存的1DCNN模型文件
model/2DCNN.h5: 保存的2DCNN模型文件

code/1DCNN.py: 1DCNN代码运行文件
code/2DCNN.py: 2DCNN代码运行文件
code/preprocessing.py: 数据预处理文件
```


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import librosa
from sklearn.model_selection import train_test_split


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 文件夹路径和txt文件路径
audio_folder = '../../data/HAD/HAD_dev/conbine/'  # 音频文件夹路径
txt_file = '../../data/HAD/HAD_dev/HAD_dev_label.txt'  # 标签文件路径

# 读取标签txt文件
df = pd.read_csv(txt_file, sep=' ', header=None, names=['filename', 'dummy', 'label'])

# 提取文件名和标签
df['file_path'] = df['filename'].apply(lambda x: os.path.join(audio_folder, x + '.wav'))

# 获取所有文件和标签
file_paths = df['file_path'].tolist()
labels = df['label'].tolist()

# 划分数据集（训练集、验证集、测试集）
x_train, X_temp, y_train, y_temp = train_test_split(file_paths, labels, test_size=0.4, random_state=42, stratify=labels)
x_valid, x_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# 定义一个函数来加载和预处理音频文件
def extract_features(file_path, fixed_height=128, fixed_width=128):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=fixed_height)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] < fixed_width:
        S_dB = np.pad(S_dB, ((0, 0), (0, fixed_width - S_dB.shape[1])), mode='constant')
    else:
        S_dB = S_dB[:, :fixed_width]

    # 可视化梅尔频谱图
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(f'Mel Spectrogram of {os.path.basename(file_path)}')
    # plt.tight_layout()
    # plt.show()

    return S_dB

# 自定义Dataset类
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        feature = extract_features(file_path)
        if self.transform:
            feature = self.transform(feature)
        return torch.tensor(feature, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

train_dataset = AudioDataset(x_train, y_train)
valid_dataset = AudioDataset(x_valid, y_valid)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=24, shuffle=False)

# 定义模型
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

model = AudioClassifier().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
best_accuracy = 0.0  # 初始化最佳准确度变量

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)  # 确保验证数据也移动到设备上
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

    val_loss /= len(valid_loader)
    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), f"../../save_model/HAD/{val_acc:.6f}%.pth")

    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)

# 准备测试数据
test_dataset = AudioDataset(x_test, np.zeros(len(x_test)))  # 假设测试集没有标签
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# 预测
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 计算准确率
correct = sum([1 for p, t in zip(predictions, true_labels) if p == t])
accuracy = correct / len(true_labels) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Epoch [1/100], Train Loss: 0.9114, Train Acc: 51.24%, Val Loss: 0.6932, Val Acc: 51.64%
# Epoch [2/100], Train Loss: 0.6494, Train Acc: 58.21%, Val Loss: 0.5905, Val Acc: 65.39%
# Epoch [3/100], Train Loss: 0.5772, Train Acc: 64.76%, Val Loss: 0.5498, Val Acc: 66.90%
# Epoch [4/100], Train Loss: 0.5620, Train Acc: 65.57%, Val Loss: 0.5457, Val Acc: 68.42%
# Epoch [5/100], Train Loss: 0.5392, Train Acc: 68.08%, Val Loss: 0.5011, Val Acc: 72.03%
# Epoch [6/100], Train Loss: 0.5160, Train Acc: 71.07%, Val Loss: 0.4806, Val Acc: 75.88%
# Epoch [7/100], Train Loss: 0.4623, Train Acc: 77.06%, Val Loss: 0.4000, Val Acc: 81.82%
# Epoch [8/100], Train Loss: 0.3493, Train Acc: 85.04%, Val Loss: 0.2276, Val Acc: 91.30%
# Epoch [9/100], Train Loss: 0.2629, Train Acc: 89.40%, Val Loss: 0.1642, Val Acc: 94.42%
# Epoch [10/100], Train Loss: 0.2250, Train Acc: 91.53%, Val Loss: 0.1245, Val Acc: 95.85%
# Epoch [11/100], Train Loss: 0.1758, Train Acc: 93.61%, Val Loss: 0.1041, Val Acc: 96.66%
# Epoch [12/100], Train Loss: 0.1752, Train Acc: 93.59%, Val Loss: 0.0926, Val Acc: 97.08%
# Epoch [13/100], Train Loss: 0.1432, Train Acc: 94.76%, Val Loss: 0.1410, Val Acc: 95.34%
# Epoch [14/100], Train Loss: 0.1348, Train Acc: 95.13%, Val Loss: 0.0792, Val Acc: 97.59%
# Epoch [15/100], Train Loss: 0.1347, Train Acc: 95.43%, Val Loss: 0.0874, Val Acc: 97.31%
# Epoch [16/100], Train Loss: 0.1197, Train Acc: 95.79%, Val Loss: 0.0822, Val Acc: 97.22%
# Epoch [17/100], Train Loss: 0.1141, Train Acc: 95.93%, Val Loss: 0.0838, Val Acc: 97.70%
# Epoch [18/100], Train Loss: 0.1085, Train Acc: 96.33%, Val Loss: 0.0907, Val Acc: 97.62%



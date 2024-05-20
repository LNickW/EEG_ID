import scipy.io
import numpy as np
import mne
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA

# 定义加载和预处理函数
def load_and_preprocess_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    # 假设.mat文件包含一个字典，其中有15个key，每个key代表一个数据段
    data = np.array([mat[key] for key in mat if not key.startswith('__')])
    data = data.reshape(data.shape[0], -1)
    return data

def preprocess_data(data):
    # 创建MNE RawArray对象
    ch_names = [f'eeg{i+1}' for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg') # 采样率为200Hz
    raw = mne.io.RawArray(data, info)

    # 滤波
    raw.filter(8., 30., fir_design='firwin')

    # 伪迹去除
    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw)
    raw = ica.apply(raw)

    return raw

def calculate_plv(epochs):
    plv_data = []
    for epoch in epochs.get_data():
        n_channels = epoch.shape[0]
        n_samples = epoch.shape[1]
        plv_matrix = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase1 = np.angle(mne.filter.hilbert(epoch[i, :]))
                phase2 = np.angle(mne.filter.hilbert(epoch[j, :]))
                plv = np.abs(np.sum(np.exp(1j * (phase1 - phase2)))) / n_samples
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv

        plv_data.append(plv_matrix.flatten())

    return np.array(plv_data)

# 加载数据
train_data, val_data, test_data = [], [], []
train_labels, val_labels, test_labels = [], [], []

# 每组数据存储在 "extracted_X_Y.mat" 文件中，其中 X 是受试者编号，Y 是数据组编号
for subject in range(1, 16):
    for group in range(1, 4):
        file_path = f'extracted_{subject}_{group}.mat'
        data = load_and_preprocess_mat(file_path)
        raw = preprocess_data(data)

        # 提取事件和标签
        events, event_ids = mne.events_from_annotations(raw)
        labels = np.array([event_ids[event[2]] for event in events])

        # 提取数据段
        epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=2, baseline=None, preload=True)

        # 提取PLV特征
        plv_features = calculate_plv(epochs)

        # 根据受试者编号划分数据集
        if subject <= 9:
            train_data.append(plv_features)
            train_labels.append(labels)
        elif 10 <= subject <= 12:
            val_data.append(plv_features)
            val_labels.append(labels)
        else:
            test_data.append(plv_features)
            test_labels.append(labels)

# 转换为 numpy 数组
train_data = np.vstack(train_data)
train_labels = np.concatenate(train_labels)
val_data = np.vstack(val_data)
val_labels = np.concatenate(val_labels)
test_data = np.vstack(test_data)
test_labels = np.concatenate(test_labels)

# 特征规范化
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# 特征降维
pca = PCA(n_components=10)
train_data = pca.fit_transform(train_data)
val_data = pca.transform(val_data)
test_data = pca.transform(test_data)

# 训练分类模型
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(train_data, train_labels)
print("分类模型训练完成")

# 交叉验证
scores = cross_val_score(mlp, train_data, train_labels, cv=5)
print('交叉验证分类准确率：', np.mean(scores))

# 绘制交叉验证结果
plt.figure(figsize=(10, 7))
sns.boxplot(data=scores)
plt.title('交叉验证分类准确率')
plt.ylabel('Accuracy')
plt.show()

# 验证模型
val_score = mlp.score(val_data, val_labels)
test_score = mlp.score(test_data, test_labels)
print('验证集分类准确率：', val_score)
print('测试集分类准确率：', test_score)

# 绘制验证集和测试集分类结果
plt.figure(figsize=(10, 7))
plt.bar(['Cross-validation', 'Validation', 'Test'], [np.mean(scores), val_score, test_score], color=['blue', 'orange', 'green'])
plt.title('分类准确率对比')
plt.ylabel('Accuracy')
plt.show()

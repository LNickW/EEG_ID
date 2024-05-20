import mne
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = 'path_to_seed_data.set'  # 假设数据格式为EEGLAB的SET格式
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# 数据预处理
def preprocess_data(raw):
    # 滤波
    raw.filter(8., 30., fir_design='firwin')  # 带通滤波，提取β频段
    print("滤波完成")

    # 伪迹去除
    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw)
    raw = ica.apply(raw)
    print("伪迹去除完成")

    return raw

raw = preprocess_data(raw)

# 绘制原始数据的时间序列图
raw.plot(n_channels=10, scalings='auto', title='原始数据的时间序列图')
plt.show()

# 提取事件和标签
events, event_ids = mne.events_from_annotations(raw)
labels = np.array([event_ids[event[2]] for event in events])
print(f"事件和标签提取完成，共提取到{len(events)}个事件")

# 绘制事件的时间序列图
mne.viz.plot_events(events, event_id=event_ids, sfreq=raw.info['sfreq'])
plt.show()

# 提取数据段
epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=2, baseline=None, preload=True)
print("数据段提取完成")

# 提取PLV特征
def calculate_plv(epochs):
    plv_data = []
    for epoch in epochs:
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

    print("PLV特征提取完成")
    return np.array(plv_data)

plv_features = calculate_plv(epochs)

# 特征规范化
scaler = StandardScaler()
plv_features = scaler.fit_transform(plv_features)
print("特征规范化完成")

# 特征降维
pca = PCA(n_components=10)
features = pca.fit_transform(plv_features)
print("特征降维完成")

# 绘制PCA结果
plt.figure(figsize=(10, 7))
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA特征分布图')
plt.colorbar()
plt.show()

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
print(f"数据集划分完成：训练集大小={len(y_train)}, 验证集大小={len(y_test)}")

# 训练分类模型
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
print("分类模型训练完成")

# 交叉验证
scores = cross_val_score(mlp, X_train, y_train, cv=5)
print('交叉验证分类准确率：', np.mean(scores))

# 绘制交叉验证结果
plt.figure(figsize=(10, 7))
sns.boxplot(scores)
plt.title('交叉验证分类准确率')
plt.ylabel('Accuracy')
plt.show()

# 验证模型
test_score = mlp.score(X_test, y_test)
print('验证集分类准确率：', test_score)

# 绘制验证集分类结果
plt.figure(figsize=(10, 7))
plt.bar(['Cross-validation', 'Test'], [np.mean(scores), test_score], color=['blue', 'green'])
plt.title('分类准确率对比')
plt.ylabel('Accuracy')
plt.show()

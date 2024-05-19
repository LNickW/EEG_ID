##use mne to preprocess the data, extract events and labels, extract epochs, extract PLV features, reduce the dimension of features using PCA, split the data into training set and validation set, train a classification model using LDA, and validate the model using cross-validation and test set.

import mne
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# 加载数据
file_path = 'path_to_seed_data.set'  # 假设数据格式为EEGLAB的SET格式
raw = mne.io.read_raw_eeglab(file_path, preload=True)


# 数据预处理
def preprocess_data(raw):
    # 滤波
    raw.filter(8., 30., fir_design='firwin')  # 带通滤波，提取β频段

    # 伪迹去除
    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw)
    ica.detect_artifacts(raw)
    raw = ica.apply(raw)

    return raw


raw = preprocess_data(raw)

# 提取事件和标签
events, event_ids = mne.events_from_annotations(raw)
labels = np.array([event_ids[event[2]] for event in events])

# 提取数据段
epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=2, baseline=None, preload=True)


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

    return np.array(plv_data)


plv_features = calculate_plv(epochs)

# 特征降维
pca = PCA(n_components=10)
features = pca.fit_transform(plv_features)

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 训练分类模型
lda = LDA()
lda.fit(X_train, y_train)

# 交叉验证
scores = cross_val_score(lda, X_train, y_train, cv=5)
print('交叉验证分类准确率：', np.mean(scores))

# 验证模型
test_score = lda.score(X_test, y_test)
print('验证集分类准确率：', test_score)

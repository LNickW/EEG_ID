import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score
import joblib


# 加载数据
def load_seed_data(file_path):
    data = sio.loadmat(file_path)
    return data['data'], data['label']


# 数据预处理：滤波
def bandpass_filter(data, low, high, fs):
    b, a = signal.butter(4, [low, high], btype='band', fs=fs)
    return signal.filtfilt(b, a, data, axis=1)


# 相位同步特征提取
def calculate_plv(eeg_data):
    n_channels = eeg_data.shape[0]
    n_samples = eeg_data.shape[1]
    plv_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase1 = np.angle(hilbert(eeg_data[i, :]))
            phase2 = np.angle(hilbert(eeg_data[j, :]))
            plv = np.abs(np.sum(np.exp(1j * (phase1 - phase2)))) / n_samples
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    return plv_matrix

# 从.mat数据中提取必要的信息
eeg_data = mat_data['djc_eeg1']

# 特征降维
def apply_pca(plv_matrix, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(plv_matrix)


# 特征分类
def classify_features(features, labels):
    lda = LDA()
    scores = cross_val_score(lda, features, labels, cv=5)
    return lda, scores


# 保存模型
def save_model(model, file_path):
    joblib.dump(model, file_path)


# 加载模型
def load_model(file_path):
    return joblib.load(file_path)


# 示例：加载数据、预处理、特征提取、降维、分类、验证
file_path = 'path_to_seed_data.mat'
eeg_data, labels = load_seed_data(file_path)
fs = 128  # 采样率

# 滤波
filtered_data = np.array([bandpass_filter(eeg, 8, 30, fs) for eeg in eeg_data])  # 例如，β频段

# 计算PLV
plv_matrices = np.array([calculate_plv(eeg) for eeg in filtered_data])

# 展平PLV矩阵
plv_features = np.array([plv.flatten() for plv in plv_matrices])

# PCA降维
features = apply_pca(plv_features, n_components=10)

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 训练分类模型
lda_model, scores = classify_features(X_train, y_train)
print('交叉验证分类准确率：', np.mean(scores))

# 保存模型
save_model(lda_model, 'lda_model.pkl')

# 加载模型
lda_model_loaded = load_model('lda_model.pkl')

# 验证模型
test_scores = lda_model_loaded.score(X_test, y_test)
print('验证集分类准确率：', test_scores)

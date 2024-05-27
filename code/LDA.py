import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, hilbert
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import os
import glob


# 滤波函数
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=-1)
    return y


# ICA去噪函数
def apply_ica(data, n_components=3):
    print(f"Before ICA, data shape: {data.shape}")
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]  # 保证数据是三维的
    n_samples, n_channels, n_times = data.shape
    reshaped_data = data.reshape(-1, n_times)
    ica = FastICA(n_components=n_components, max_iter=1000, tol=0.001, random_state=42)
    cleaned_data = ica.fit_transform(reshaped_data)
    cleaned_data = cleaned_data.reshape(n_samples, n_channels, -1)
    print(f"After ICA, data shape: {cleaned_data.shape}")
    return cleaned_data


# 计算PLV特征函数
def compute_plv(data):
    print(f"Before PLV, data shape: {data.shape}")
    if len(data.shape) != 3:
        raise ValueError("Input data must be 3-dimensional.")
    n_samples, n_channels, n_times = data.shape
    plv = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = np.angle(hilbert(data[:, i, :])) - np.angle(hilbert(data[:, j, :]))
            plv[i, j] = np.abs(np.sum(np.exp(1j * phase_diff), axis=-1)) / n_times

    print(f"After PLV, data shape: {plv.shape}")
    return plv


# PCA降维函数
def apply_pca(data, n_components=10):
    print(f"Before PCA, data shape: {data.shape}")
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data.reshape(data.shape[0], -1))
    print(f"After PCA, data shape: {reduced_data.shape}")
    return reduced_data


# 训练LDA模型并进行交叉验证
def train_lda(X, y):
    lda = LDA()
    skf = StratifiedKFold(n_splits=10)
    scores = cross_val_score(lda, X, y, cv=skf)
    print(f'Cross-validation accuracy: {np.mean(scores):.2f}')
    return lda.fit(X, y)


# 加载.mat文件数据
def load_mat_files(directory):
    files = glob.glob(os.path.join(directory, '*.mat'))
    all_data = []

    for file in files:
        mat_data = scipy.io.loadmat(file)
        for key in mat_data:
            if isinstance(mat_data[key], np.ndarray) and mat_data[key].dtype == 'float64':
                all_data.append(mat_data[key])

    return all_data


# 划分数据集
def split_dataset(data, labels, train_size=0.6, test_size=0.2, val_size=0.2):
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, train_size=train_size, random_state=42)
    test_ratio = test_size / (test_size + val_size)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=test_ratio, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val


# 预处理和特征提取的函数
def preprocess_and_extract_features(data_list, lowcut, highcut, fs, n_components=10):
    filtered_data_list = [bandpass_filter(data, lowcut, highcut, fs) for data in data_list]
    cleaned_data_list = [apply_ica(data) for data in filtered_data_list]
    plv_features_list = [compute_plv(data) for data in cleaned_data_list]
    reduced_data_list = [apply_pca(data, n_components=n_components) for data in plv_features_list]
    return reduced_data_list


# 主函数
if __name__ == "__main__":
    directory = 'D:\PyCharm\EEG_ID\dataset'  # 替换为实际.mat文件目录
    raw_data_list = load_mat_files(directory)

    # 创建标签（假设每个文件有一个对应的标签）
    labels = np.array([i // 15 for i in range(len(raw_data_list))])

    # 划分数据集
    X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(raw_data_list, labels)

    # 预处理
    fs = 250  # 采样频率
    lowcut = 8.0
    highcut = 30.0

    # 对训练数据进行滤波、去噪、特征提取和降维
    reduced_data_train = preprocess_and_extract_features(X_train, lowcut, highcut, fs)
    reduced_data_train = np.vstack(reduced_data_train)  # 将列表中的数据垂直堆叠成矩阵

    # 对测试数据进行相同的处理
    reduced_data_test = preprocess_and_extract_features(X_test, lowcut, highcut, fs)
    reduced_data_test = np.vstack(reduced_data_test)

    # 对验证数据进行相同的处理
    reduced_data_val = preprocess_and_extract_features(X_val, lowcut, highcut, fs)
    reduced_data_val = np.vstack(reduced_data_val)

    # 训练LDA模型并进行交叉验证
    lda_model = train_lda(reduced_data_train, y_train)

    # 测试和验证
    y_test_pred = lda_model.predict(reduced_data_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test accuracy: {test_accuracy:.2f}')

    y_val_pred = lda_model.predict(reduced_data_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation accuracy: {val_accuracy:.2f}')

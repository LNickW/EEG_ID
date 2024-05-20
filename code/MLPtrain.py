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
import os
from loader import load_mat_files  # 如果需要，可以更改为正确的导入方式

# 定义加载和预处理函数
def load_and_preprocess_mat(file_path):
    try:
        if os.path.exists(file_path):
            print(f"Loading file: {file_path}")
            mat = scipy.io.loadmat(file_path)
            data_list = [mat[key] for key in mat if not key.startswith('__')]

            # 确保所有数组的形状一致
            max_len = max([d.shape[1] for d in data_list])
            uniform_data = []
            for d in data_list:
                if d.shape[1] < max_len:
                    # 使用零填充以确保所有数组长度相同
                    padded_d = np.pad(d, ((0, 0), (0, max_len - d.shape[1])), 'constant')
                else:
                    padded_d = d
                uniform_data.append(padded_d)

            data = np.array(uniform_data)
            data = data.reshape(data.shape[0], -1)
            print(f"Loaded data shape: {data.shape}")
            return data
        else:
            raise FileNotFoundError(f"File not found: {file_path}")  # 文件不存在时抛出异常
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# 定义数据预处理函数
def preprocess_data(data, sfreq):
    try:
        # 创建通道名称
        ch_names = [f'eeg{i+1}' for i in range(data.shape[0])]
        # 创建 MNE 信息对象
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        # 创建原始 MNE 数据对象
        raw = mne.io.RawArray(data, info)

        # EEG 信号滤波
        raw.filter(8., 30., fir_design='firwin')

        # 使用 ICA 进行信号去噪
        ica = ICA(n_components=15, random_state=97)
        ica.fit(raw)
        raw = ica.apply(raw)

        return raw
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None

# 计算 PLV 特征
def calculate_plv(epochs):
    plv_data = []
    for epoch in epochs.get_data():
        n_channels = epoch.shape[0]
        n_samples = epoch.shape[1]
        plv_matrix = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # 计算相位差
                phase1 = np.angle(mne.filter.hilbert(epoch[i, :]))
                phase2 = np.angle(mne.filter.hilbert(epoch[j, :]))
                # 计算 PLV
                plv = np.abs(np.sum(np.exp(1j * (phase1 - phase2)))) / n_samples
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv

        plv_data.append(plv_matrix.flatten())

    return np.array(plv_data)

# 加载数据
train_data, val_data, test_data = [], [], []
train_labels, val_labels, test_labels = [], [], []

sfreq = 200  # 设置实际采样率为200Hz
data_directory = 'D:\\PyCharm\\EEG_ID\\dataset'  # 替换为实际的数据目录路径

all_data = load_mat_files(data_directory)

for subject in range(1, 16):
    for group in range(1, 4):
        file_path = os.path.join(data_directory, f'extracted_{subject}_{group}.mat')
        data = load_and_preprocess_mat(file_path)
        if data is not None:
            raw = preprocess_data(data, sfreq)

            if raw is not None:  # 确保数据预处理成功
                events, event_ids = mne.events_from_annotations(raw)
                labels = np.array([event_ids[event[2]] for event in events])

                epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=2, baseline=None, preload=True)

                plv_features = calculate_plv(epochs)

                if subject <= 9:
                    train_data.append(plv_features)
                    train_labels.append(labels)
                elif 10 <= subject <= 12:
                    val_data.append(plv_features)
                    val_labels.append(labels)
                else:
                    test_data.append(plv_features)
                    test_labels.append(labels)

# 合并数据和标签
if train_data:
    train_data = np.vstack(train_data)
    train_labels = np.concatenate(train_labels)
if val_data:
    val_data = np.vstack(val_data)
    val_labels = np.concatenate(val_labels)
if test_data:
    test_data = np.vstack(test_data)
    test_labels = np.concatenate(test_labels)

# 特征标准化
try:
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
except Exception as e:
    print(f"Error in feature scaling: {e}")

# 使用 PCA 进行降维
try:
    pca = PCA(n_components=10)
    train_data = pca.fit_transform(train_data)
    val_data = pca.transform(val_data)
    test_data = pca.transform(test_data)
except Exception as e:
    print(f"Error in PCA: {e}")

# 创建 MLP 分类器模型
try:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp.fit(train_data, train_labels)
    print("分类模型训练完成")
except Exception as e:
    print(f"Error in MLP training: {e}")

# 交叉验证评估模型性能
try:
    scores = cross_val_score(mlp, train_data, train_labels, cv=5)
    print('交叉验证分类准确率：', np.mean(scores))
except Exception as e:
    print(f"Error in cross-validation: {e}")

# 绘制交叉验证结果的箱线图
try:
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=scores)
    plt.title('交叉验证分类准确率')
    plt.ylabel('Accuracy')
    plt.show()
except Exception as e:
    print(f"Error in plotting cross-validation results: {e}")

# 在验证集和测试集上评估模型性能
try:
    val_score = mlp.score(val_data, val_labels)
    test_score = mlp.score(test_data, test_labels)
    print('验证集分类准确率：', val_score)
    print('测试集分类准确率：', test_score)
except Exception as e:
    print(f"Error in evaluating model performance: {e}")

# 绘制分类准确率对比图
try:
    plt.figure(figsize=(10, 7))
    plt.bar(['Cross-validation', 'Validation', 'Test'], [np.mean(scores), val_score, test_score], color=['blue', 'orange', 'green'])
    plt.title('分类准确率对比')
    plt.ylabel('Accuracy')
    plt.show()
except Exception as e:
    print(f"Error in plotting accuracy comparison: {e}")


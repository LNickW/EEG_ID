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
from loader import load_mat_files  # 确保你正确导入 load_mat_files


def extract_data_and_annotations(struct):
    """
    递归提取结构体中的 `data` 和 `annotations` 字段。
    """
    data = None
    annotations = None
    if isinstance(struct, np.ndarray) and struct.size == 1:
        struct = struct[0]
    if isinstance(struct, np.void) and struct.dtype.names:
        if 'data' in struct.dtype.names and 'annotations' in struct.dtype.names:
            data = struct['data']
            annotations = struct['annotations']
        else:
            for name in struct.dtype.names:
                if data is None or annotations is None:
                    extracted_data, extracted_annotations = extract_data_and_annotations(struct[name])
                    if extracted_data is not None:
                        data = extracted_data
                    if extracted_annotations is not None:
                        annotations = extracted_annotations
    return data, annotations


def load_and_preprocess_mat(file_path):
    try:
        if os.path.exists(file_path):
            print(f"Loading file: {file_path}")
            mat = scipy.io.loadmat(file_path)

            data_list = []
            annotations_list = []
            for key in mat:
                if not key.startswith('__'):
                    item = mat[key]
                    print(f"Key: {key}, Type: {type(item)}, Shape: {item.shape}")
                    data, annotations = extract_data_and_annotations(item)
                    if data is not None and annotations is not None:
                        print(f"Extracted data shape from {key}: {data.shape}")
                        data_list.append(data)
                        annotations_list.append(annotations)
                    else:
                        print(f"Warning: 'data' or 'annotations' field not found in struct {key}")

            if not data_list:
                raise ValueError("No valid 'data' fields found in the .mat file.")

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
            return data, annotations_list
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def preprocess_data(data, sfreq):
    """
    预处理数据，创建MNE RawArray对象
    """
    try:
        ch_names = [f'eeg{i+1}' for i in range(data.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        raw.filter(8., 30., fir_design='firwin')

        ica = ICA(n_components=15, random_state=97)
        ica.fit(raw)
        raw = ica.apply(raw)
        return raw
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return None

# 计算 PLV 特征
def calculate_plv(epochs):
    try:
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
    except Exception as e:
        print(f"Error calculating PLV: {e}")
        return None

# 加载数据
train_data, val_data, test_data = [], [], []
train_labels, val_labels, test_labels = [], [], []

sfreq = 200  # 设置实际采样率为200Hz
data_directory = 'D:\PyCharm\EEG_ID\dataset_2'  # 替换为实际的数据目录路径

all_data = load_mat_files(data_directory)

for subject in range(1, 16):
    for group in range(1, 4):
        file_path = os.path.join(data_directory, f'processed_extracted_{subject}_{group}.mat')
        data = load_and_preprocess_mat(file_path)
        if data is not None:
            raw = preprocess_data(data, sfreq)
            if raw is not None:  # 检查数据预处理是否成功
                # 检查原始数据中的注释
                annotations = raw.annotations
                if len(annotations) == 0:
                    print(f"No annotations found in file: {file_path}")
                    continue
                else:
                    print(f"Found annotations: {annotations}")

                # 提取事件
                events, event_ids = mne.events_from_annotations(raw)
                if len(events) == 0:  # 检查是否提取到事件
                    print(f"No events found in file: {file_path}")
                    continue
                else:
                    print(f"Found events: {events}")

                labels = np.array([event_ids[event[2]] for event in events])

                # 创建 Epochs 对象
                try:
                    epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=2, baseline=None, preload=True)
                    plv_features = calculate_plv(epochs)
                except Exception as e:
                    print(f"Error creating Epochs or calculating PLV for file {file_path}: {e}")
                    continue

                # 将数据和标签分别加入对应的列表
                if subject <= 9:
                    train_data.append(plv_features)
                    train_labels.append(labels)
                elif 10 <= subject <= 12:
                    val_data.append(plv_features)
                    val_labels.append(labels)
                else:
                    test_data.append(plv_features)
                    test_labels.append(labels)

if train_data:
    train_data = np.vstack(train_data)
    train_labels = np.concatenate(train_labels)
else:
    print("No training data found!")

if val_data:
    val_data = np.vstack(val_data)
    val_labels = np.concatenate(val_labels)
else:
    print("No validation data found!")

if test_data:
    test_data = np.vstack(test_data)
    test_labels = np.concatenate(test_labels)
else:
    print("No test data found!")

if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
    print("Not enough data to proceed with model training and evaluation.")
else:
    # 数据标准化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    # 使用 PCA 降维
    pca = PCA(n_components=10)
    train_data = pca.fit_transform(train_data)
    val_data = pca.transform(val_data)
    test_data = pca.transform(test_data)

    # 训练分类模型
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp.fit(train_data, train_labels)
    print("分类模型训练完成")

    # 交叉验证分类准确率
    try:
        scores = cross_val_score(mlp, train_data, train_labels, cv=5)
        print('交叉验证分类准确率：', np.mean(scores))
        plt.figure(figsize=(10, 7))
        sns.boxplot(data=scores)
        plt.title('交叉验证分类准确率')
        plt.ylabel('Accuracy')
        plt.show()
    except Exception as e:
        print(f"Error in cross-validation: {e}")

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

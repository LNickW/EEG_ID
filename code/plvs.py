import os
import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# 带通滤波器
def butter_bandpass(lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=1)
    return y

# 进行ICA去伪影
def remove_artifacts_ica(data, n_components=14, max_iter=200, tol=0.001):
    ica = FastICA(n_components=n_components, whiten='unit-variance', max_iter=max_iter, tol=tol)
    transformed_data = ica.fit_transform(data.T).T
    return transformed_data

# 计算PLV特征
def compute_plv(data, window_size):
    n_samples, n_channels = data.shape
    n_windows = n_samples // window_size
    if n_windows == 0:
        print("Error: Not enough data for one window.")
        return None

    plv_matrix = np.zeros((n_windows, n_channels, n_channels))

    for i in range(n_windows):
        segment = data[i * window_size:(i + 1) * window_size, :]
        complex_signal = np.exp(1j * np.angle(segment))
        for j in range(n_channels):
            for k in range(j, n_channels):
                plv_matrix[i, j, k] = np.abs(np.mean(complex_signal[:, j] * np.conj(complex_signal[:, k])))
    return plv_matrix

# 保存PLV矩阵和生成热力图
def save_plv_matrix_and_plot(plv_matrix, mat_file, array_output_dir, heatmap_output_dir):
    if plv_matrix is None:
        print("No PLV matrix to save or plot.")
        return

    n_windows, n_channels, _ = plv_matrix.shape
    plv_avg = np.zeros((n_channels, n_channels))

    for j in range(n_channels):
        for k in range(j, n_channels):
            plv_avg[j, k] = np.mean(plv_matrix[:, j, k])

    base_filename = os.path.splitext(os.path.basename(mat_file))[0]

    array_output_path = os.path.join(array_output_dir, f"{base_filename}_plvavg.npy")
    np.save(array_output_path, plv_avg)

    plt.figure(figsize=(10, 8))
    plt.imshow(plv_avg, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('PLV Average Matrix Heatmap')
    plt.xlabel('Channels')
    plt.ylabel('Channels')
    heatmap_output_path = os.path.join(heatmap_output_dir, f"{base_filename}_plvavg_heatmap.png")
    plt.savefig(heatmap_output_path)
    plt.close()

# 主函数
def main(input_folder, array_output_dir, heatmap_output_dir, lowcut=8, highcut=50, fs=200, window_size=100):
    mat_files = [file for file in os.listdir(input_folder) if file.endswith('.mat')]

    for mat_file in mat_files:
        mat_path = os.path.join(input_folder, mat_file)
        mat_data = sio.loadmat(mat_path)
        if 'concatenated_matrix' in mat_data:
            data = mat_data['concatenated_matrix']
        else:
            print(f"Error: 'concatenated_matrix' not found in {mat_file}")
            continue

        if data.shape[0] != 14:
            print(f"Error: Unexpected data shape {data.shape}, expected (14, n_samples)")
            continue

        data = data.T

        print(f"Processing {mat_file}...")
        filtered_data = bandpass_filter(data, lowcut, highcut, fs)
        clean_data = remove_artifacts_ica(filtered_data)
        plv_matrix = compute_plv(filtered_data, window_size)

        save_plv_matrix_and_plot(plv_matrix, mat_file, array_output_dir, heatmap_output_dir)
        print(f"{mat_file} processed.")

# 参数配置
input_folder = r'D:\PyCharm\EEG_ID\pre_data'  # 修改为包含MAT文件的文件夹路径
array_output_dir = r'D:\PyCharm\EEG_ID\numpydata'  # 修改为保存数组的文件夹路径
heatmap_output_dir = r'D:\PyCharm\EEG_ID\imgs'  # 修改为保存热力图的文件夹路径

# 运行主函数
main(input_folder, array_output_dir, heatmap_output_dir)

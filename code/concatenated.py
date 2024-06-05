import os
import scipy.io as sio
import numpy as np

# 定义基础目录
base_dir = 'D:\PyCharm\EEG_ID\data_14'
save_dir = 'D:\PyCharm\EEG_ID\pre_data'

# 遍历基础目录下的所有 .mat 文件
for file_name in os.listdir(base_dir):
    if file_name.endswith('.mat') and file_name != 'label.mat':
        file_path = os.path.join(base_dir, file_name)

        # 加载 .mat 文件
        mat_data = sio.loadmat(file_path)

        # 初始化一个列表来存储所有矩阵
        all_matrices = []

        # 遍历 .mat 文件中的所有键值对
        for key in mat_data:
            if isinstance(mat_data[key], np.ndarray):
                matrix = mat_data[key]
                if matrix.ndim == 2:  # 确保这是一个二维矩阵
                    all_matrices.append(matrix)

        # 将所有矩阵按列连接在一起
        if all_matrices:
            concatenated_matrix = np.concatenate(all_matrices, axis=1)

            # 保存连接后的数据为新的 .mat 文件
            save_path = os.path.join(save_dir, f'{file_name}')
            sio.savemat(save_path, {'concatenated_matrix': concatenated_matrix})

            print(f'Processed {file_path}, saved concatenated data to {save_path}')

import os
import scipy.io as sio
import numpy as np

# 定义基础目录
base_dir = 'D:\SEED\SEED_EEG\Preprocessed_EEG'
output_dir = '/dataset'

# 遍历基础目录下的所有 .mat 文件
for file_name in os.listdir(base_dir):
    if file_name.endswith('.mat') and file_name != 'label.mat':
        file_path = os.path.join(base_dir, file_name)

        # 加载 .mat 文件
        mat_data = sio.loadmat(file_path)

        # 初始化一个字典来存储提取后的数据
        extracted_data = {}

        # 遍历 .mat 文件中的所有键值对
        for key in mat_data:
            if isinstance(mat_data[key], np.ndarray):
                matrix = mat_data[key]
                if matrix.ndim == 2:  # 确保这是一个二维矩阵
                    # 提取前三个通道的数据
                    extracted_matrix = matrix[:3, :]
                    extracted_data[key] = extracted_matrix

        # 保存提取后的数据为新的 .mat 文件
        save_path = os.path.join(output_dir, f'extracted_{file_name}')
        sio.savemat(save_path, extracted_data)

        print(f'Processed {file_path}, saved extracted data to {save_path}')

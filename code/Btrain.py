import os
import numpy as np

# 文件夹路径
input_folder_path = r"D:\PyCharm\EEG_ID\avg"
output_folder_path = r"D:\PyCharm\EEG_ID\btrain"
output_file_name = "B_train.npy"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 用于存储所有向量的列表
all_vectors = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(input_folder_path):
    if file_name.endswith('.npy'):  # 检查文件是否是.npy文件
        # 构造文件的完整路径
        file_path = os.path.join(input_folder_path, file_name)
        # 读取文件
        matrix = np.load(file_path)
        # 提取上三角元素，不包括对角线
        upper_triangle_indices = np.triu_indices(14, k=1)
        vector = matrix[upper_triangle_indices]
        # 将向量添加到列表
        all_vectors.extend(vector.tolist())

# 将列表转换为列向量
B_train = np.array(all_vectors).reshape(-1, 1)

print("B_train shape:", B_train.shape)

# 保存B_train到指定文件夹
output_file_path = os.path.join(output_folder_path, output_file_name)
np.save(output_file_path, B_train)

print(f"B_train saved to {output_file_path}")

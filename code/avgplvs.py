import numpy as np

# 读取三个numpy文件
file1 = r"D:\PyCharm\EEG_ID\numpydata\15_20130709_plvavg.npy"
file2 = r"D:\PyCharm\EEG_ID\numpydata\15_20131016_plvavg.npy"
file3 = r"D:\PyCharm\EEG_ID\numpydata\15_20131105_plvavg.npy"

matrix1 = np.load(file1)
matrix2 = np.load(file2)
matrix3 = np.load(file3)

# 计算三个numpy矩阵的平均值
average_matrix = (matrix1 + matrix2 + matrix3) / 3

# 保存平均值至指定文件夹
output_file = r"D:\PyCharm\EEG_ID\numpydata\avg15.npy"
np.save(output_file, average_matrix)

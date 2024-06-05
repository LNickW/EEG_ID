import numpy as np
from sklearn.decomposition import PCA

# 读取列向量B_train
B_train = np.load(r"D:\PyCharm\EEG_ID\btrain\B_train.npy")

# 创建PCA对象
pca = PCA()

# 拟合PCA模型并进行降维
Y_train = pca.fit_transform(B_train)

# 输出降维后的特征向量矩阵形状
print("Y_train shape:", Y_train.shape)

# 可选：输出PCA的解释方差比例
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 保存特征向量空间V
V = pca.components_
np.save("D:\PyCharm\EEG_ID\ytrain_V\PCA_feature_space.npy", V)
print("V shape:", V.shape)
# 保存降维后的特征向量矩阵Y_train
np.save("D:\PyCharm\EEG_ID\ytrain_V\Y_train.npy", Y_train)

print("Feature space V saved to PCA_feature_space.npy")
print("Y_train saved to Y_train.npy")

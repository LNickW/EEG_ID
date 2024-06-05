import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 读取降维后的特征向量矩阵 Y_train
Y_train = np.load("D:\PyCharm\EEG_ID\ytrain_V\Y_train.npy")

# 根据需要加载训练数据的标签
labels = np.load("D:\PyCharm\EEG_ID\lablenp\PLV_labels.npy")

# 检查 labels 的形状
print("Labels shape:", labels.shape)

# 如果 labels 是二维的，将其转换为一维数组
if len(labels.shape) > 1:
    labels = labels.ravel()

# 创建LDA对象
lda = LinearDiscriminantAnalysis()

# 训练LDA模型
T_train = lda.fit_transform(Y_train, labels)

# 保存训练好的LDA模型
np.save("LDA_model.npy", lda)

# 可视化分类结果
plt.figure(figsize=(10, 6))
for label in np.unique(labels):
    plt.scatter(T_train[labels == label, 0],np.zeros_like(T_train[labels == label, 0]), label=label)
plt.title('LDA Classification')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.grid(True)
plt.savefig("LDA_classification.png")
plt.show()

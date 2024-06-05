import numpy as np

# 加载PLV数据
plv_data = np.load(r"D:\PyCharm\EEG_ID\btrain\B_train.npy")

# 确定每个人的PLV数据长度
num_channels = 14
plv_length_per_person = num_channels * (num_channels - 1) // 2  # 上三角矩阵长度

# 确定每个人的数据范围并为每个PLV矩阵打上对应的标签
all_plvs = []
labels = []

for person_id in range(15):
    start_index = person_id * plv_length_per_person
    end_index = (person_id + 1) * plv_length_per_person
    person_plv_data = plv_data[start_index:end_index]
    all_plvs.append(person_plv_data)
    labels.extend([person_id] * plv_length_per_person)

# 将列表转换为numpy数组
all_plvs = np.array(all_plvs)
labels = np.array(labels)
label = labels.T
# 保存标签数据到文件
np.save("D:\PyCharm\EEG_ID\lablenp\PLV_labels.npy", label)

print("Labels generated and saved to PLV_labels.npy file.")

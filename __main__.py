import scipy.io
from scipy.io import loadmat
import mne
import matplotlib.pyplot as plt


# 读取.mat文件
mat_data = scipy.io.loadmat(r'D:\SEED\SEED_EEG\Preprocessed_EEG\1_20131027.mat')

# 从.mat数据中提取必要的信息
eeg_data = mat_data['djc_eeg1']


sfreq = 256

ch_names = ['EEG{}'.format(i) for i in range(1, 63)]  # 假设你的数据有62个通道
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# 创建原始MNE对象
raw = mne.io.RawArray(eeg_data, info)

# 绘制原始数据
raw.plot(n_channels=62, scalings='auto', title='EEG Data')
plt.show()

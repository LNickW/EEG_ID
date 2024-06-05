import os
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmaps(directory, rows, cols):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5)
    for i in range(rows):
        for j in range(cols):
            file_num = i * cols + j + 1
            file_name = os.path.join(directory, f'avg{file_num}.npy')
            if os.path.exists(file_name):
                matrix = np.load(file_name)
                ax = axs[i, j]
                im = ax.imshow(matrix, cmap='jet', interpolation='nearest')
                ax.set_title(f'Subject {file_num}')
                fig.colorbar(im, ax=ax)
            else:
                ax = axs[i, j]
                ax.axis('off')  # 如果文件不存在，关闭当前子图
    plt.show()

if __name__ == "__main__":
    directory = r'D:\PyCharm\EEG_ID\avg'
    rows = 3
    cols = 5
    plot_heatmaps(directory, rows, cols)


import os
import numpy as np
import scipy.io
import mne


def load_and_process_mat(file_path, sfreq):
    try:
        # 加载MAT文件
        mat_data = scipy.io.loadmat(file_path)

        # 查找包含'data'和'annotations'字段的结构体
        for key in mat_data:
            if not key.startswith('__'):
                if 'data' in mat_data[key].dtype.names and 'annotations' in mat_data[key].dtype.names:
                    data = mat_data[key]['data'][0, 0]
                    annotations_struct = mat_data[key]['annotations'][0, 0]

                    # 提取注释字段
                    onset = annotations_struct['onset'][0]
                    duration = annotations_struct['duration'][0]
                    description = [str(desc[0]) for desc in annotations_struct['description'][0]]

                    # 创建MNE注释对象
                    annotations = mne.Annotations(onset=onset, duration=duration, description=description)

                    # 创建MNE信息对象
                    ch_names = [f'eeg{i + 1}' for i in range(data.shape[0])]
                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

                    # 创建MNE RawArray对象
                    raw = mne.io.RawArray(data, info)
                    raw.set_annotations(annotations)

                    return raw, key
        return None, None
    except Exception as e:
        print(f"Error loading and processing MAT file {file_path}: {e}")
        return None, None


def save_processed_data(raw, output_file_path, key):
    try:
        # 提取数据和注释
        data = raw.get_data()
        annotations = raw.annotations

        # 创建保存结构
        mat_data = {
            key: {
                'data': data,
                'annotations': {
                    'onset': annotations.onset,
                    'duration': annotations.duration,
                    'description': np.array(annotations.description, dtype=object)
                }
            }
        }

        # 保存到MAT文件
        scipy.io.savemat(output_file_path, mat_data)
        print(f"Successfully saved processed data to {output_file_path}")
    except Exception as e:
        print(f"Error saving processed data to {output_file_path}: {e}")


# 示例使用
sfreq = 200  # 采样率
input_data_directory = 'D:\PyCharm\EEG_ID\dataset'
output_data_directory = 'D:\PyCharm\EEG_ID\dataset_2'

# 确保输出目录存在
if not os.path.exists(output_data_directory):
    os.makedirs(output_data_directory)

for file_name in os.listdir(input_data_directory):
    if file_name.endswith('.mat'):
        input_file_path = os.path.join(input_data_directory, file_name)
        output_file_path = os.path.join(output_data_directory, f"processed_{file_name}")

        raw, key = load_and_process_mat(input_file_path, sfreq)
        if raw is not None and key is not None:
            save_processed_data(raw, output_file_path, key)
        else:
            print(f"Failed to process {input_file_path}")

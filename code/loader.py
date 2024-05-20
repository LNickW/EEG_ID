import scipy.io
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def load_mat_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".mat"):
            file_path = os.path.join(directory, filename)
            try:
                print(f"Loading file: {file_path}")
                mat_contents = scipy.io.loadmat(file_path)

                # Initialize a list to collect valid data from the current file
                file_data = []

                for key in mat_contents:
                    if not key.startswith("__"):  # Ignore default keys added by scipy.io.loadmat
                        data = mat_contents[key]

                        # Check if data has shape (3, any number)
                        if data.shape[0] == 3:
                            file_data.append(data)
                        else:
                            print(f"Incorrect shape in file {file_path}, key {key}: {data.shape}")

                # If all keys have valid shapes, append the file data to all_data
                if len(file_data) == 15:
                    all_data.append(file_data)
                else:
                    print(f"File {file_path} does not contain 15 valid keys.")

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    return all_data


if __name__ == "__main__":
    directory = 'D:\\PyCharm\\EEG_ID\\dataset'
    all_data = load_mat_files(directory)

    if all_data:
        # Combine all data into a single numpy array
        combined_data = np.concatenate([np.concatenate(file_data, axis=1) for file_data in all_data], axis=1)
        print(f"Combined data shape: {combined_data.shape}")

        # Example data transformation with StandardScaler
        scaler = StandardScaler()
        transformed_data = scaler.fit_transform(combined_data.T).T  # Transpose for sklearn, then transpose back
        print(f"Transformed data shape: {transformed_data.shape}")

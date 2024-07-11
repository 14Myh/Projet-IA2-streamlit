import os
import cv2
import numpy as np
from descriptor import glcm, bitdesc

def extract_features(image_path, descriptor):
    img = cv2.imread(image_path, 0)
    if img is not None:
        features = descriptor(image_path)
        return features
    else:
        print(f"Warning: Unable to read image at {image_path}")
        return None

# browse all folders and subfolders and save as a signature (glcm and bit)
def process_dataset(root_folder, descriptor_func, descriptor_name):
    all_features = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, root_folder)
                folder_name = os.path.basename(os.path.dirname(image_path))
                features = descriptor_func(image_path)
                if features is not None:
                    features = features + [folder_name, relative_path]
                    print(relative_path)
                    all_features.append(features)
    if all_features:
        signatures = np.array(all_features)
        np.save(f'{descriptor_name}_signatures.npy', signatures)
        print(f'Successfully stored {descriptor_name} signatures!')
    else:
        print(f"No features extracted for {descriptor_name}")

def main():
    base_directory = './Dataset'
    process_dataset(base_directory, glcm, 'glcm')
    process_dataset(base_directory, bitdesc, 'bit')

if __name__ == '__main__':
    main()

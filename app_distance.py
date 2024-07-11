#app_distance
import os
from descriptor import glcm
from distances import manhattan, euclidean, chebyshev, canberra

# browse all folders and subfolders
def process_images_in_directory(directory):
    image_features = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file)
                features = glcm(image_path)
                image_features[file] = features

    image_files = list(image_features.keys())
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            file1, file2 = image_files[i], image_files[j]
            feat1, feat2 = image_features[file1], image_features[file2]
            print(f'Comparing {file1} and {file2}:')
            print(f'Manhattan: {manhattan(feat1, feat2)}')
            print(f'Euclidean: {euclidean(feat1, feat2)}')
            print(f'Chebyshev: {chebyshev(feat1, feat2)}')
            print(f'Canberra: {canberra(feat1, feat2)}')
            print('-' * 40)

def main():
    base_directory = './Dataset'
    process_images_in_directory(base_directory)

if __name__ == '__main__':
    main()

#app
import os
from descriptor import glcm, bitdesc

def process_images_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file)
                feat_glcm = glcm(image_path)
                feat_bit = bitdesc(image_path)
                print(f'Image: {file}')
                print(f'GLCM\n-----\n{feat_glcm}')
                print(f'BiT\n---\n{feat_bit}')
                print('-' * 40)

def main():
    base_directory = './Dataset'
    process_images_in_directory(base_directory)

if __name__ == '__main__':
    main()

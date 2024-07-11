from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo
import cv2
import numpy as np

# glcm descriptor
def glcm(image_path):
    data = cv2.imread(image_path, 0)
    if data is None:
        raise ValueError(f"Image at path {image_path} could not be read")
    co_matrix = graycomatrix(data, [1], [np.pi/4], symmetric=False, normed=False)
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    contrast = graycoprops(co_matrix, 'contrast')[0, 0]
    correlation = graycoprops(co_matrix, 'correlation')[0, 0]
    energy = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homogeneity = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [dissimilarity, contrast, correlation, energy, asm, homogeneity]

# BiT descriptor
def bitdesc(image_path):
    data = cv2.imread(image_path, 0)
    if data is None:
        raise ValueError(f"Image at path {image_path} could not be read")
    return bio_taxo(data)

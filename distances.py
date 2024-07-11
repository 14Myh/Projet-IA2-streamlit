import numpy as np
from scipy.spatial import distance

# calculate the distance according to choice
def manhattan(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return np.sum(np.abs(v1 - v2))

def euclidean(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return np.sqrt(np.sum((v1 - v2) ** 2))

def chebyshev(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return np.max(np.abs(v1 - v2))

def canberra(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return distance.canberra(v1, v2)

def retrieve_similar_image(features_db, query_features, distance_type, num_results):
    distances = []
    for instance in features_db:
        features, label, img_path = instance[:-2], instance[-2], instance[-1]
        if distance_type == 'manhattan':
            dist = manhattan(query_features, features)
        elif distance_type == 'euclidean':
            dist = euclidean(query_features, features)
        elif distance_type == 'chebyshev':
            dist = chebyshev(query_features, features)
        elif distance_type == 'canberra':
            dist = canberra(query_features, features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]

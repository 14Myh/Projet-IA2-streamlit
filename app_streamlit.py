import streamlit as st
import numpy as np
from descriptor import glcm, bitdesc
from distances import manhattan, euclidean, chebyshev, canberra
import os

BASE_DIR = os.path.abspath('Dataset') # base 

def load_signatures(file_path):
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

def compare_images(features_db, query_features, distance_type):
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
    return distances

st.title(':violet[Content-Based Image Retrieval - (CBIR)]')

# upload an image type .jpg .jpeg .png
st.write(':red[Ensure image paths are accessible for display.]')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.sidebar:
        # descriptor
        descriptor_option = st.selectbox(
            'Choose descriptor',
            ('GLCM', 'BiT')
        )

        # distance
        distance_option = st.selectbox(
            'Choose distance measure',
            ('Manhattan', 'Euclidean', 'Chebyshev', 'Canberra')
        )

        num_results = st.slider('Number of similar images to display', 1, 20, 5)

    if st.button('Find Similar Images'):
        if descriptor_option == 'GLCM':
            features = glcm("temp.jpg")
            signatures = load_signatures('glcm_signatures.npy')
        else:
            features = bitdesc("temp.jpg")
            signatures = load_signatures('bit_signatures.npy')

        if signatures.size > 0:
            st.write('Features extracted:')
            st.write(features)

            distance_function = distance_option.lower()
            results = compare_images(signatures, features, distance_function)

            st.write(f'Top {num_results} matches:')

            for img_path, dist, label in results[:num_results]:
                abs_img_path = os.path.join(BASE_DIR, img_path)
                relative_img_path = os.path.relpath(abs_img_path, BASE_DIR)  # relative image
                if os.path.isfile(abs_img_path):
                    st.write(f"Image: {relative_img_path}, \nDistance: {dist}")
                    st.image(abs_img_path, width=200)
                else:
                    st.write(f"Image: {relative_img_path}, \nDistance: {dist}")
                    st.warning(f"Image at path {relative_img_path} could not be found.")

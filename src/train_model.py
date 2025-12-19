import numpy as np
import joblib
import pickle
import os
from typing import Tuple, List, Union
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

image_size = 100*100

def print_mock_data(mock_face_data, centered_face_data, mean_image, mock_labels):
    print(mock_face_data)
    print("---------------------")
    print(centered_face_data)
    print("---------------------")
    print(mean_image)
    print("---------------------")
    print(mock_labels)

#saves the models we created and trained in train_model()
def save_models(mean_image: np.ndarray, pca: PCA, knn: KNeighborsClassifier):
    os.makedirs("model_files", exist_ok=True)

    #saving mean_image to 'model_files/mean_image.pj1'
    with open('model_files/mean_image.pk1', 'wb') as f:
        pickle.dump(mean_image, f)

    #saving pca to 'model_files/mean_image.pj1'
    joblib.dump(pca, 'model_files/pca_model.pk1')

    with open('model_files/knn_classifier.pk1', 'wb')as f:
        pickle.dump(knn, f)

    print("saved all models to model_files directory")

#trains the model using PCA and knn (input is number of dimensions for PCA, and k neighbors)
def train_model(
    face_data: np.ndarray,
    labels: np.ndarray,
    num_reduced_dimensions: int,
    num_neighors: int):

    #centering data around the mean image (PCA STEP 1)
    mean_image = np.mean(face_data, axis=0)
    centered_face_data = face_data - mean_image

    #implementation of PCA
    pca = PCA(num_reduced_dimensions)

    features = pca.fit_transform(centered_face_data)

    #implementation of knn

    knn = KNeighborsClassifier(num_neighors)

    knn.fit(features, labels)

    print("just trained knn classifier")

    save_models(mean_image, pca, knn)
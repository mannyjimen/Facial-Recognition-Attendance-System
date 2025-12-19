import numpy as np
import os
import pickle
import joblib

from typing import Optional
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#global model variables

global_mean_image: Optional[np.ndarray] = None
global_pca_model: Optional[PCA] = None
global_knn_classifier: Optional[KNeighborsClassifier ]= None

#loading models from data folder
def load_recognition_models(model_dir: str = 'model_files') -> bool:
    global global_mean_image, global_pca_model, global_knn_classifier

    mean_image_path = os.path.join(model_dir,'mean_image.pk1')
    pca_model_path = os.path.join(model_dir, 'pca_model.pk1')
    knn_classifier_path = os.path.join(model_dir, 'knn_classifier.pk1')

    try:
        with open(mean_image_path, 'rb') as f:
            global_mean_image = pickle.load(f)

        global_pca_model = joblib.load(pca_model_path)

        with open(knn_classifier_path, 'rb') as f:
            global_knn_classifier = pickle.load(f)

        print("recognition models loaded successfully")
        return True

    except FileNotFoundError:
        print(f"file not found in '{model_dir}'")
        return False
    except Exception as e:
        print(f"unexpected error occurred during model loading: {e}")
        return False

def recognize_face(face_array: np.ndarray) -> str:
    global global_mean_image, global_pca_model, global_knn_classifier

    #safety check here MAYBE

    flattened_face = face_array.flatten().reshape(1, -1)
    centered_face = flattened_face - global_mean_image
    projected_feature = global_pca_model.transform(centered_face)
    predicted_name = global_knn_classifier.predict(projected_feature)[0]

    return predicted_name
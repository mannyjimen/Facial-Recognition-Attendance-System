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
def train_model(folder_path: str, num_reduced_dimensions: int, num_neighors: int):
    #mocking data for training
    num_people = 3 #me, junyi, sammy
    num_images_per_person = 5
    num_images = num_people * num_images_per_person
    
    #creating mock data
    mock_face_data = np.random.rand(num_images, image_size)

    #creating mock lables
    labels = ['Manny', 'Sammy', 'Junyi']
    mock_labels = np.array([label for label in labels for _ in range(num_images_per_person)])

    #centering data around the mean image (PCA STEP 1)
    mean_image = np.mean(mock_face_data, axis=0)
    centered_face_data = mock_face_data - mean_image

    #printing mock data
    # print_mock_data(mock_face_data, centered_face_data, mean_image, mock_labels)

    #implementation of PCA
    pca = PCA(num_reduced_dimensions)

    features = pca.fit_transform(centered_face_data)

    #implementation of knn

    knn = KNeighborsClassifier(num_neighors)

    knn.fit(features, mock_labels)

    print("just trained knn classifier")

    save_models(mean_image, pca, knn)

#returns name of recognized face
def recognize_face(face_array: np.ndarray) -> str:
    return ""

train_model("test", 5, 4)
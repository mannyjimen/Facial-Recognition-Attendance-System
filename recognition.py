import numpy as np
from typing import Tuple, List, Union

image_size = 100*100

#returns data from folder into a np.ndarray, and the labels in an np.ndarray
def load_data(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return ""

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

    print(mock_face_data)
    print("---------------------")
    print(centered_face_data)
    print("---------------------")
    print(mean_image)
    print("---------------------")
    print(mock_labels)

#returns name of recognized face
def recognize_face(face_array: np.ndarray) -> str:
    return ""

train_model("test", 5, 4)
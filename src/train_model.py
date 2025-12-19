# --- START OF FILE train_model.py ---
import numpy as np
import joblib
import pickle
import os
import cv2
from typing import Tuple, List, Union
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

PROCESSED_DATA_DIR = os.path.join("..", "data", "processed")

def save_models(mean_image: np.ndarray, pca: PCA, knn: KNeighborsClassifier):
    os.makedirs("model_files", exist_ok=True)

    with open('model_files/mean_image.pk1', 'wb') as f:
        pickle.dump(mean_image, f)

    joblib.dump(pca, 'model_files/pca_model.pk1')

    with open('model_files/knn_classifier.pk1', 'wb') as f:
        pickle.dump(knn, f)

    print("Saved all models to 'model_files' directory")

def train_model(face_data: np.ndarray, labels: np.ndarray, num_reduced_dimensions: int, num_neighbors: int):
    print(f"Training with {len(face_data)} images and {num_reduced_dimensions} dimensions...")

    # 1. Centering data
    mean_image = np.mean(face_data, axis=0)
    centered_face_data = face_data - mean_image

    # 2. PCA
    # Ensure n_components isn't larger than the number of samples
    n_components = min(num_reduced_dimensions, len(face_data))
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(centered_face_data)

    # 3. KNN
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(features, labels)

    print("KNN classifier trained.")
    save_models(mean_image, pca, knn)

def load_training_data(data_dir):
    """
    Loads images from subfolders, flattens them, and creates labels.
    """
    face_vectors = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return np.array([]), np.array([])

    persons = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for person_name in persons:
        person_dir = os.path.join(data_dir, person_name)
        files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]

        for f in files:
            path = os.path.join(person_dir, f)
            # Read in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Flatten 100x100 -> 10000
                face_vectors.append(img.flatten())
                labels.append(person_name)

    return np.array(face_vectors), np.array(labels)

def main():
    # Load data
    print("Loading data...")
    X, y = load_training_data(PROCESSED_DATA_DIR)

    if len(X) == 0:
        print("No training data found. Run collect_images.py and preprocess.py first.")
        return

    # Hyperparameters
    # 50 dimensions is usually enough for faces, 5 neighbors is standard
    num_dims = 50
    k_neighbors = 5

    train_model(X, y, num_dims, k_neighbors)

if __name__ == "__main__":
    main()

import numpy as np
from typing import Tuple, List, Union
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

image_size = 100*100

def recognize_face(face_array: np.ndarray) -> str:
    return ""
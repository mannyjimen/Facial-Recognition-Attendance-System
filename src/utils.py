# src/utils.py
import os
import cv2
import numpy as np
from typing import List, Tuple

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_subfolders(path: str) -> List[str]:
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def load_images_from_folder(folder: str) -> List[np.ndarray]:
    images = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def get_face_detector():
    # Haar cascade face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load Haar cascade. Check OpenCV installation.")
    return detector

def detect_largest_face(gray_img: np.ndarray, detector, scaleFactor=1.3, minNeighbors=5):
    faces = detector.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    if len(faces) == 0:
        return None
    # Choose the largest face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return (x, y, w, h)

def preprocess_face_from_frame(frame: np.ndarray, detector, size: Tuple[int, int]=(100, 100)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_box = detect_largest_face(gray, detector)
    if face_box is None:
        return None
    x, y, w, h = face_box
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, size)
    return face_resized, (x, y, w, h)

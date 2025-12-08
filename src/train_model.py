# src/train_model.py
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from utils import list_subfolders, load_images_from_folder

PROC_DATA_DIR = os.path.join("..", "data", "processed")
MODELS_DIR = os.path.join("..", "models")

N_COMPONENTS = 100  # PCA components

def load_dataset():
    X = []
    y = []
    persons = list_subfolders(PROC_DATA_DIR)
    if not persons:
        raise RuntimeError("No processed data found. Run preprocess.py first.")

    for person in persons:
        folder = os.path.join(PROC_DATA_DIR, person)
        imgs = load_images_from_folder(folder)
        for img in imgs:
            X.append(img.flatten())
            y.append(person)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Loaded {X.shape[0]} images, each of dimension {X.shape[1]}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # PCA
    print("Fitting PCA...")
    pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=42)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Classifier (SVM)
    print("Training SVM classifier...")
    clf = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    clf.fit(X_train_pca, y_train)

    # Evaluation
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save models
    pca_path = os.path.join(MODELS_DIR, "pca_model.pkl")
    clf_path = os.path.join(MODELS_DIR, "classifier.pkl")
    le_path  = os.path.join(MODELS_DIR, "label_encoder.pkl")

    joblib.dump(pca, pca_path)
    joblib.dump(clf, clf_path)
    joblib.dump(le, le_path)

    print(f"\nSaved PCA model to {pca_path}")
    print(f"Saved classifier to {clf_path}")
    print(f"Saved label encoder to {le_path}")

if __name__ == "__main__":
    main()

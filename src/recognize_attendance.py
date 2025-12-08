# src/recognize_attendance.py
import os
import cv2
import csv
import datetime
import numpy as np
import joblib

from utils import get_face_detector, preprocess_face_from_frame

MODELS_DIR = os.path.join("..", "models")
ATT_DIR    = os.path.join("..", "attendance")
TARGET_SIZE = (100, 100)

def load_models():
    pca_path = os.path.join(MODELS_DIR, "pca_model.pkl")
    clf_path = os.path.join(MODELS_DIR, "classifier.pkl")
    le_path  = os.path.join(MODELS_DIR, "label_encoder.pkl")

    if not (os.path.exists(pca_path) and os.path.exists(clf_path) and os.path.exists(le_path)):
        raise RuntimeError("Models not found. Run train_model.py first.")

    pca = joblib.load(pca_path)
    clf = joblib.load(clf_path)
    le  = joblib.load(le_path)
    return pca, clf, le

def get_attendance_file():
    os.makedirs(ATT_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()
    fname = f"attendance_{today}.csv"
    fpath = os.path.join(ATT_DIR, fname)
    if not os.path.exists(fpath):
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time", "confidence"])
    return fpath

def append_attendance(name: str, confidence: float, att_file: str):
    now = datetime.datetime.now()
    date_str = now.date().isoformat()
    time_str = now.strftime("%H:%M:%S")
    with open(att_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str, f"{confidence:.3f}"])

def main():
    pca, clf, le = load_models()
    detector = get_face_detector()
    att_file = get_attendance_file()

    # Keep track of who has already been logged in this session
    seen_names = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot access webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        result = preprocess_face_from_frame(frame, detector, TARGET_SIZE)
        if result is not None:
            face_resized, (x, y, w, h) = result

            # Flatten and project through PCA
            face_vec = face_resized.flatten().astype(np.float32).reshape(1, -1)
            face_pca = pca.transform(face_vec)

            # Predict
            proba = clf.predict_proba(face_pca)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
            name = le.inverse_transform([pred_idx])[0]

            # Threshold for accepting recognition
            THRESH = 0.6  # tune as needed
            if confidence >= THRESH:
                label = f"{name} ({confidence:.2f})"
                # Log attendance if seeing for first time this session
                if name not in seen_names:
                    append_attendance(name, confidence, att_file)
                    seen_names.add(name)
            else:
                label = "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended. Attendance saved.")

if __name__ == "__main__":
    main()

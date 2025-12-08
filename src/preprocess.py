# src/preprocess.py
import os
import cv2
from utils import ensure_dir, list_subfolders, get_face_detector, detect_largest_face

RAW_DATA_DIR = os.path.join("..", "data", "raw")
PROC_DATA_DIR = os.path.join("..", "data", "processed")
TARGET_SIZE = (100, 100)

def preprocess_person(person_name: str, detector):
    src_dir = os.path.join(RAW_DATA_DIR, person_name)
    dst_dir = os.path.join(PROC_DATA_DIR, person_name)
    ensure_dir(dst_dir)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    count = 0

    for fname in files:
        fpath = os.path.join(src_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = detect_largest_face(gray, detector)
        if box is None:
            print(f"No face found in {fpath}, skipping.")
            continue

        x, y, w, h = box
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, TARGET_SIZE)
        out_path = os.path.join(dst_dir, fname)
        cv2.imwrite(out_path, face_resized)
        count += 1

    print(f"{person_name}: processed {count} images.")

def main():
    detector = get_face_detector()
    persons = list_subfolders(RAW_DATA_DIR)
    if not persons:
        print("No subfolders found in data/raw. Run collect_images.py first.")
        return

    ensure_dir(PROC_DATA_DIR)

    for person_name in persons:
        preprocess_person(person_name, detector)

if __name__ == "__main__":
    main()

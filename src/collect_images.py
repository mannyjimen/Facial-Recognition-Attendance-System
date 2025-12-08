# src/collect_images.py
import os
import cv2
from utils import ensure_dir

RAW_DATA_DIR = os.path.join("..", "data", "raw")

def main():
    person_name = input("Enter person name (folder name): ").strip()
    if not person_name:
        print("Name cannot be empty.")
        return

    person_dir = os.path.join(RAW_DATA_DIR, person_name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot access webcam.")
        return

    print("Press SPACE to capture an image, 'q' to quit.")
    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Collecting Images - " + person_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            img_path = os.path.join(person_dir, f"{img_count:03d}.jpg")
            cv2.imwrite(img_path, frame)
            img_count += 1
            print(f"Saved {img_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {img_count} images for {person_name} in {person_dir}")

if __name__ == "__main__":
    main()

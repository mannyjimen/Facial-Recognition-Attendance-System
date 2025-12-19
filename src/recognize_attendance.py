import cv2
import os
import csv
from datetime import datetime
from utils import get_face_detector, preprocess_face_from_frame
import recognition

ATTENDANCE_FILE = "attendance.csv"

def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    
    name_list = []
    if file_exists:
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) == 3:
                    entry_name, entry_time, entry_date = line
                    if entry_date == date_string:
                        name_list.append(entry_name)
    
    if name not in name_list:
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Time', 'Date'])
            
            writer.writerow([name, time_string, date_string])
            print(f"Attendance marked for: {name}")

def main():
    if not recognition.load_recognition_models(model_dir='model_files'):
        print("Failed to load models.")
        return

    detector = get_face_detector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        result = preprocess_face_from_frame(frame, detector, size=(100, 100))

        if result is not None:
            face_resized, (x, y, w, h) = result

            try:
                predicted_name = recognition.recognize_face(face_resized)

                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.rectangle(display_frame, (x, y-35), (x+w, y), (0, 255, 0), cv2.FILLED)
                cv2.putText(display_frame, predicted_name, (x+6, y-6), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                mark_attendance(predicted_name)

            except Exception as e:
                print(e)

        cv2.imshow('Attendance System', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

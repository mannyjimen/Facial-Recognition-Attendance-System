# Facial Recognition Attendance System

A Python-based attendance system that uses Computer Vision to detect and recognize faces. The system captures student images, trains a K-Nearest Neighbors (KNN) classifier using PCA for dimensionality reduction, and logs attendance into a CSV file.

## Features
* **Face Detection:** Uses Haar Cascades to locate faces in real-time.
* **Face Recognition:** Implements PCA (Principal Component Analysis) and KNN (K-Nearest Neighbors) for accurate identification.
* **Automated Pipeline:** Includes a script to streamline data collection, preprocessing, and model training.
* **Attendance Logging:** Automatically generates CSV reports with time stamps.

## Installation

### 1. Prerequisites
* Python 3.9 or higher
* A working webcam

### 2. Clone the Repository

3. Install Dependencies
Critical: This project requires specific versions of Numpy to maintain compatibility with OpenCV.

pip install -r requirements.txt

How to use:
Go to terminal
1. cd src
2. python3 run_pipeline.py
3. take a few photos by pressing spacebar and press q to close camera
4. python recognize_attendance.py

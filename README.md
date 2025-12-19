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

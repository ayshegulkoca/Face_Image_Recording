# 📸 Face Image Recording, Training, and Attendance System

This repository contains a complete Python project for collecting facial image data, training a face recognition model, and utilizing it for applications like automated attendance tracking, all wrapped in a Graphical User Interface (GUI).

The project leverages the power of **OpenCV** for camera interaction and image processing, and **MediaPipe** for high-accuracy face detection and landmark tracking, ensuring high-quality, aligned data for machine learning tasks.

## 🌟 Features

* **Robust Data Collection:** Records cropped and eye-aligned face images, along with rich metadata and raw MediaPipe landmarks.
* **Face Alignment:** Automatically rotates and scales captured faces so that the eyes are consistently horizontal, significantly improving data quality for model training.
* **Model Training:** Dedicated script (`face_trainer.py`) for training a face recognition model on the collected dataset.
* **Real-time Recognition:** Script (`face_recognizer.py`) for live face detection and identification.
* **Attendance Tracking:** Outputs a log (`attendance.csv`/`.xlsx`) for recognized individuals.
* **Graphical User Interface (GUI):** A user-friendly interface (`gui.py`) to manage all main functions.

## 🛠️ Prerequisites

Before running any script, ensure you have Python (version 3.7+) installed.

### 1. Installation

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv

# 2. Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install the required packages
pip install opencv-python mediapipe numpy pandas

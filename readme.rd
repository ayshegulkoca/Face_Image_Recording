📸 Face Data Recorder Project

This project uses the computer's camera to record face data, including cropped, aligned images, and MediaPipe facial landmark metadata, for use in machine learning or computer vision tasks.

It uses OpenCV for camera interaction and image processing, and MediaPipe for high-accuracy face detection and landmark tracking.

🛠️ Prerequisites

Before running the scripts, ensure you have Python (3.7+) installed and the necessary libraries.

1. Installation

It is highly recommended to use a virtual environment.

# Create a virtual environment (optional but recommended)
python -m venv venv
# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install opencv-python mediapipe numpy


🚀 Usage

The primary script for data collection is face_data_recorder.py.

1. Run the Script

python face_data_recorder.py


2. Controls and Interaction

Once the camera window opens, use the following keyboard shortcuts to control the recording process.

Key

Action

Description

n

Set Label/Name

(MANDATORY FIRST STEP) Prompts you in the terminal to enter a label (e.g., user_john or happy_set). This creates the storage folder for the data.

s

Toggle Saving

Starts or stops the automatic continuous saving of face images. Data is saved every SAVE_EVERY_N_FRAMES (default: 1).

c

Toggle Center-Only

When enabled, only saves faces that are near the center of the camera frame, ensuring better framing.

l

Toggle Landmarks

Shows or hides the MediaPipe Face Mesh landmarks overlay on the live feed.

m

Toggle Use Mesh

Toggles the use of the Face Mesh model (which is required for eye-based alignment).

q

Quit & Finalize

Closes the camera and writes the final manifest files (manifest.csv and manifest.json).

📁 Output Data Structure

The script automatically creates a root folder named dataset and organizes the data within it.

dataset/
├── [your_label_name]/
│   ├── image_filename.jpg                 (Aligned face image)
│   ├── image_filename.json                (Per-image metadata: bbox, score, etc.)
│   └── image_filename.landmarks.json      (Raw MediaPipe landmarks for the image)
├── manifest.csv                           (CSV summary of all recorded data)
└── manifest.json                          (JSON array of all recorded metadata)


Key Data Features

Aligned Faces: The script uses facial landmarks (specifically eye positions) to rotate and scale the cropped face so that the eyes are always horizontal and in a consistent position. This significantly improves data quality for training models.

Metadata: Each saved image is accompanied by a .json file containing its bounding box, detection confidence score, and a timestamp, providing rich, easily digestible training data.

📝 Script Overview

face_data_recorder.py

This is the main, robust data collection tool. It implements:

Face Detection (mp.solutions.face_detection).

Face Mesh (mp.solutions.face_mesh) for fine-grained landmarks.

Eye Alignment logic using geometric transformation (align_face).

Data Management: Saves images, landmarks, and creates two master manifest files.

face.py (Simple Cropping)

This script is a simplified example that only performs basic face detection and saves unaligned, raw face crops into the recorded_faces folder. It is useful for quick testing but is generally superseded by face_data_recorder.py for professional data collection.
import cv2
import os
import pickle
import numpy as np
import time
from face_data_recorder import save_attendance

# ================= CONFIG =================
TRAINER_PATH = 'trainer'
TRAINER_FILE = 'trainer.yml'
LABELS_FILE = 'labels.pickle'
# LBPH için yüksek değer = daha az katı tanıma (daha çok tanıma)
CONFIDENCE_THRESHOLD = 85

# --- Load Model and Labels ---
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(TRAINER_PATH, TRAINER_FILE))
    print("✅ Recognizer model loaded.")
except Exception:
    print("❌ Error loading model. Please run face_trainer.py first.")
    # Uygulamanın model olmadan çalışmasını engellemek için çıkış
    # exit()

try:
    with open(os.path.join(TRAINER_PATH, LABELS_FILE), 'rb') as f:
        id_to_name = pickle.load(f)
    print("✅ Labels loaded.")
except Exception:
    print("❌ Error loading labels. Please run face_trainer.py first.")
    # exit()

# Load cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ================= RECOGNITION AND ATTENDANCE LOGIC =================

def start_recognition():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    # Set of names for whom attendance has been taken in this session (to avoid duplicates)
    attended_today = set()

    print("\n--- Starting Face Recognition Mode ---")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        img = cv2.flip(img, 1)  # Mirror image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Recognize the face using the trained model
            id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Default to Unknown
            name = "Unknown"
            color = (0, 0, 255)  # Red for Unknown

            if confidence < CONFIDENCE_THRESHOLD:  # LBPH: lower confidence = better match
                # Face Recognized
                name = id_to_name.get(id_, "Unknown ID")
                confidence_text = f"({round(confidence)}%)"
                color = (0, 255, 0)  # Green for Known

                # --- ATTENDANCE LOGIC ---
                if name != "Unknown ID" and name not in attended_today:
                    save_attendance(name)
                    attended_today.add(name)
                    print(f"ATTENDANCE TAKEN for: {name}")

            else:
                # Face Not Recognized (Unknown Person)
                confidence_text = f"({round(confidence)}%)"
                name = "Unknown Person"

                # --- NEW PERSON LOGIC ---
                cv2.putText(img, "NEW PERSON! Register via GUI", (x - 20, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw rectangle and name/confidence
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img, confidence_text, (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face Recognition Attendance - Press Q to Quit', img)

        # --- KAMERA ÇIKIŞ DÜZELTMESİ: 'q' tuşuna basıldığında döngüyü kır ---
        k = cv2.waitKey(10) & 0xff
        if k == ord('q'):
            break

    # Kamera serbest bırakma ve pencereleri kapatma
    cam.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")


if __name__ == "__main__":
    start_recognition()
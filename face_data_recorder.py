import cv2
import mediapipe as mp
import os
import sys
from datetime import datetime
from openpyxl import Workbook, load_workbook
import time

# ================= CONFIG =================
OUTPUT_DIR = "recorded_faces"
ATTENDANCE_FILE = "attendance.xlsx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= STUDENT NAME =================
if len(sys.argv) > 1:
    student_name = sys.argv[1]
else:
    student_name = "Unknown"

# ================= ATTENDANCE =================
def save_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        wb = load_workbook(ATTENDANCE_FILE)
    else:
        wb = Workbook()
        wb.remove(wb.active)

    if today in wb.sheetnames:
        ws = wb[today]
    else:
        ws = wb.create_sheet(today)
        ws.append(["Name", "Time"])

    names = [row[0] for row in ws.iter_rows(min_row=2, values_only=True)]
    if name not in names:
        ws.append([name, time_now])
        wb.save(ATTENDANCE_FILE)

# ================= CAMERA (HIDDEN) =================
def run_camera_hidden(name):
    mp_face_detection = mp.solutions.face_detection

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera could not be opened")
        return

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    ) as face_detection:

        start_time = time.time()
        face_saved = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    face = frame[y:y + bh, x:x + bw]

                    if face.size != 0 and not face_saved:
                        img_path = os.path.join(
                            OUTPUT_DIR, f"{name}.jpg"
                        )
                        cv2.imwrite(img_path, face)
                        save_attendance(name)
                        face_saved = True
                        print(f"✅ Face saved for {name}")
                        break

            # 🕒 Güvenlik: max 5 saniye kamera açık kalsın
            if face_saved or (time.time() - start_time > 5):
                break

    # ===== CLEAN EXIT =====
    cap.release()
    cv2.destroyAllWindows()
    print("✔ Camera closed (hidden mode)")

# ================= MAIN =================
if __name__ == "__main__":
    print(f"Starting hidden attendance for: {student_name}")
    run_camera_hidden(student_name)

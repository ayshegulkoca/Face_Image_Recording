import cv2
import mediapipe as mp
import os
from openpyxl import Workbook, load_workbook
from datetime import datetime

# ================= CONFIG =================
OUTPUT_DIR = "recorded_faces"
ATTENDANCE_FILE = "attendance.xlsx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= STUDENT NAME =================
student_name = input("Enter student name: ").strip()

# ================= MEDIAPIPE =================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ================= ATTENDANCE FUNCTION =================
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        wb = load_workbook(ATTENDANCE_FILE)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.append(["Name", "Date", "Time"])

    ws.append([name, date, time])
    wb.save(ATTENDANCE_FILE)

attendance_marked = False
img_count = 0

# ================= MAIN LOOP =================
with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
) as face_detection:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                face_crop = frame[y:y + bh, x:x + bw]

                if face_crop.size != 0:
                    filename = f"{OUTPUT_DIR}/{student_name}_{img_count}.jpg"
                    cv2.imwrite(filename, face_crop)
                    img_count += 1

                mp_drawing.draw_detection(frame, detection)

                if not attendance_marked:
                    mark_attendance(student_name)
                    attendance_marked = True
                    print(f"Attendance marked for {student_name}")

        cv2.imshow("Student Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Attendance saved in:", ATTENDANCE_FILE)


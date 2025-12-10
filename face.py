import cv2
import mediapipe as mp
import os

# Create output folder
OUTPUT_DIR = "recorded_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                # Get face bounding box
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)

                # Crop face
                face_crop = frame[y:y+h_box, x:x+w_box]

                # Save cropped face
                if face_crop.size != 0:
                    filename = f"{OUTPUT_DIR}/face_{img_count}.jpg"
                    cv2.imwrite(filename, face_crop)
                    img_count += 1

                # Draw box on screen
                mp_drawing.draw_detection(frame, detection)

        cv2.imshow("Face Recorder", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Dataset saved in:", OUTPUT_DIR)

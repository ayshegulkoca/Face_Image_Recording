import cv2
import mediapipe as mp
import numpy as np
import os
import json
import csv
from datetime import datetime

# -------------------- Configuration --------------------
OUTPUT_ROOT = "dataset"
IMAGE_FORMAT = ".jpg"  # or .png
DESIRED_FACE_SIZE = (256, 256)
SAVE_EVERY_N_FRAMES = 1  # when saving mode is on
MIN_DETECTION_CONFIDENCE = 0.5
CENTER_THRESHOLD = 0.25  # fraction of frame center distance allowed to save when center_only

# -------------------- Helpers --------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image(path, img):
    cv2.imwrite(path, img)


def landmarks_to_list(landmarks, image_w, image_h):
    """Convert MediaPipe normalized landmarks to pixel (x,y,z) list."""
    out = []
    for lm in landmarks:
        out.append((lm.x * image_w, lm.y * image_h, lm.z * max(image_w, image_h)))
    return out


def compute_eye_centers(landmark_list):
    """Return left and right eye center (x,y) using common FaceMesh indices."""
    # Using typical FaceMesh indices for eye outer/inner corners and pupils approximations
    # left eye landmarks (approx): 33, 133; right eye: 362, 263
    try:
        left = np.mean([landmark_list[33][:2], landmark_list[133][:2]], axis=0)
        right = np.mean([landmark_list[362][:2], landmark_list[263][:2]], axis=0)
        return tuple(left), tuple(right)
    except Exception:
        return None, None


def align_face(img, left_eye, right_eye, output_size=DESIRED_FACE_SIZE, offset_pct=(0.35, 0.35)):
    """Align face so that eyes are horizontal and scaled to output_size.
    offset_pct = fraction of output image where eyes should be placed (x_pct, y_pct)
    """
    # Convert to numpy
    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)

    # compute angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # distance between eyes
    dist = np.linalg.norm(right_eye - left_eye)

    # desired distance between eyes in output
    desired_dist = (1.0 - 2 * offset_pct[0]) * output_size[0]
    scale = desired_dist / (dist + 1e-6)

    # eyes center
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # shift to put eyes at desired position
    tx = output_size[0] * 0.5 - eyes_center[0]
    ty = output_size[1] * offset_pct[1] - eyes_center[1]
    M[0, 2] += tx
    M[1, 2] += ty

    aligned = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_CUBIC)
    return aligned

# -------------------- Main Recorder Class --------------------

class FaceRecorder:
    def __init__(self, camera_idx=0):
        self.cap = cv2.VideoCapture(camera_idx)
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, refine_landmarks=True, min_detection_confidence=MIN_DETECTION_CONFIDENCE)
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()

        self.saving = False
        self.label = "unknown"
        self.center_only = False
        self.show_landmarks = True
        self.use_mesh = True

        self.frame_idx = 0
        ensure_dir(OUTPUT_ROOT)
        self.manifest = []

    def prompt_label(self):
        new_label = input("Enter new label/name (folder will be created): ").strip()
        if new_label:
            self.label = new_label
            ensure_dir(os.path.join(OUTPUT_ROOT, self.label))
            print(f"Label set to: {self.label}")

    def save_metadata(self, out_path, meta):
        # append to manifest
        self.manifest.append(meta)
        # write per-image json
        with open(out_path + ".json", 'w') as f:
            json.dump(meta, f, indent=2)

    def write_manifest(self):
        if not self.manifest:
            return
        csv_path = os.path.join(OUTPUT_ROOT, 'manifest.csv')
        json_path = os.path.join(OUTPUT_ROOT, 'manifest.json')
        # CSV
        keys = list(self.manifest[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=keys)
            writer.writeheader()
            for row in self.manifest:
                writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        # JSON
        with open(json_path, 'w') as jf:
            json.dump(self.manifest, jf, indent=2)
        print(f"Wrote manifest ({len(self.manifest)} entries) to {csv_path} and {json_path}")

    def process(self):
        print("Starting Face Recorder. Press 'n' to set name, 's' to toggle saving, 'q' to quit.")
        save_counter = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera read failed")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = self.face_detection.process(rgb)
            mesh_results = None
            if self.use_mesh:
                mesh_results = self.face_mesh.process(rgb)

            # draw detections
            if detections and detections.detections:
                for det in detections.detections:
                    self.drawing_utils.draw_detection(frame, det)

            # draw mesh overlay
            if self.show_landmarks and mesh_results and mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec)

            # Handle saving logic
            to_save = []  # list of dicts: {image, bbox, landmarks_pixels}
            if detections and detections.detections:
                for i, det in enumerate(detections.detections):
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    # clamp
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w, x + bw)
                    y2 = min(h, y + bh)

                    crop = frame[y1:y2, x1:x2].copy()

                    landmarks_px = None
                    left_eye = None
                    right_eye = None
                    if mesh_results and mesh_results.multi_face_landmarks and i < len(mesh_results.multi_face_landmarks):
                        lm = mesh_results.multi_face_landmarks[i].landmark
                        landmarks_px = landmarks_to_list(lm, w, h)
                        left_eye, right_eye = compute_eye_centers(landmarks_px)

                    to_save.append({
                        'crop': crop,
                        'bbox': (x1, y1, x2, y2),
                        'landmarks': landmarks_px,
                        'left_eye': left_eye,
                        'right_eye': right_eye,
                        'detection_score': det.score[0] if det.score else None
                    })

            # Save images according to mode
            if self.saving and (self.frame_idx % SAVE_EVERY_N_FRAMES == 0):
                for face_idx, info in enumerate(to_save):
                    # check center-only if required
                    can_save = True
                    if self.center_only:
                        x1, y1, x2, y2 = info['bbox']
                        face_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        frame_center = (w / 2.0, h / 2.0)
                        d = np.linalg.norm(np.array(face_center) - np.array(frame_center))
                        maxd = np.linalg.norm(np.array([w * 0.5, h * 0.5]))
                        if (d / (maxd + 1e-6)) > CENTER_THRESHOLD:
                            can_save = False

                    if can_save:
                        label_dir = os.path.join(OUTPUT_ROOT, self.label)
                        ensure_dir(label_dir)

                        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
                        fname = f"{self.label}_{timestamp}_{face_idx}{IMAGE_FORMAT}"
                        out_path = os.path.join(label_dir, fname)

                        # if we have eye coords, align; else save bbox crop resized
                        if info['left_eye'] and info['right_eye']:
                            aligned = align_face(frame, info['left_eye'], info['right_eye'], output_size=DESIRED_FACE_SIZE)
                            save_image(out_path, aligned)
                            saved_shape = aligned.shape
                        else:
                            resized = cv2.resize(info['crop'], DESIRED_FACE_SIZE)
                            save_image(out_path, resized)
                            saved_shape = resized.shape

                        meta = {
                            'filename': out_path,
                            'label': self.label,
                            'timestamp': datetime.utcnow().isoformat() + 'Z',
                            'bbox': info['bbox'],
                            'detection_score': info['detection_score'],
                            'landmarks_present': info['landmarks'] is not None,
                        }

                        # save landmarks CSV/JSON if present
                        if info['landmarks']:
                            lm_path = out_path + '.landmarks.json'
                            with open(lm_path, 'w') as lmf:
                                json.dump(info['landmarks'], lmf)
                            meta['landmarks_file'] = lm_path

                        # write per-image metadata
                        self.save_metadata(out_path, meta)
                        save_counter += 1

            # Draw small HUD
            hud = f"Label: {self.label} | Saving: {self.saving} | Center-only: {self.center_only} | Mesh: {self.use_mesh} | Landmarks: {self.show_landmarks} | Saved: {len(self.manifest)}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('FaceRecorder', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('n'):
                # prompt in terminal (blocking)
                self.prompt_label()
            elif key == ord('s'):
                self.saving = not self.saving
                print(f"Saving toggled: {self.saving}")
            elif key == ord(' '):
                # single shot save: toggle saving on for one frame
                original = self.saving
                self.saving = True
                # force immediate save on this frame
                # handled by logic above since frame_idx used
                # restore after a small delay (we'll set saving False next loop)
                self.saving = False
            elif key == ord('c'):
                self.center_only = not self.center_only
                print(f"Center-only toggled: {self.center_only}")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"Show landmarks toggled: {self.show_landmarks}")
            elif key == ord('m'):
                self.use_mesh = not self.use_mesh
                print(f"Use Mesh toggled: {self.use_mesh}")

            self.frame_idx += 1

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        self.write_manifest()


if __name__ == '__main__':
    fr = FaceRecorder()
    fr.process()

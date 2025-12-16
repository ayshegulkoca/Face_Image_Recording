import cv2
import numpy as np
from PIL import Image
import os
import pickle

# ================= CONFIG =================
PATH = 'recorded_faces'
TRAINER_PATH = 'trainer'
TRAINER_FILE = 'trainer.yml'

os.makedirs(TRAINER_PATH, exist_ok=True)

# LBPHFaceRecognizer modelini başlat
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_images_and_labels(path):
    face_samples = []
    ids = []

    # recorded_faces içindeki tüm klasör adlarını (öğrenci isimleri) al
    user_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # ID'leri isimlerle eşleştirme sözlüğü oluştur
    name_to_id = {name: i for i, name in enumerate(user_names)}

    if not user_names:
        print("❌ Error: No student folders found in 'recorded_faces'. Please record data first.")
        return face_samples, ids

    print("--- Collecting Data and Assigning IDs ---")
    for user_name, current_id in name_to_id.items():
        print(f"Assigned ID {current_id} to user: {user_name}")

        user_folder = os.path.join(path, user_name)

        # Kullanıcının klasöründeki tüm resim dosyalarını al
        user_image_paths = [os.path.join(user_folder, f) for f in os.listdir(user_folder) if f.endswith('.jpg')]

        for image_path in user_image_paths:
            try:
                # Görüntüyü gri tonlamalı olarak yükle
                pil_img = Image.open(image_path).convert('L')
                img_numpy = np.array(pil_img, 'uint8')

                # Yüz tespiti yap
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    # Yüz örneğini ve etiketini (ID) listelere ekle
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(current_id)

            except Exception as e:
                # Dosya okuma hatalarını görmezden gel
                pass

                # ID-Name eşleştirmesini kaydet (Tanıma için lazım)
    id_to_name = {v: k for k, v in name_to_id.items()}
    with open(os.path.join(TRAINER_PATH, 'labels.pickle'), 'wb') as f:
        pickle.dump(id_to_name, f)
        print("\n✅ ID-Name mapping saved to labels.pickle")

    return face_samples, ids


# --- Start Training ---
faces, ids = get_images_and_labels(PATH)

if len(faces) > 0:
    print(f"\n--- Training Model Started with {len(faces)} samples from {len(np.unique(ids))} unique users ---")
    recognizer.train(faces, np.array(ids))

    # Save the model
    recognizer.write(os.path.join(TRAINER_PATH, TRAINER_FILE))

    print(f"\n✅ Model trained. Model saved to: {os.path.join(TRAINER_PATH, TRAINER_FILE)}")
else:
    print("\n❌ Training failed. No valid face samples found.")
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_SCRIPT_PATH = os.path.join(BASE_DIR, "face_data_recorder.py")
RECOGNITION_SCRIPT_PATH = os.path.join(BASE_DIR, "face_recognizer.py")
TRAINER_SCRIPT_PATH = os.path.join(BASE_DIR, "face_trainer.py")


def start_attendance_recording():
    """Starts camera to record face data for a new/existing student."""
    name = entry.get().strip()

    if not name:
        messagebox.showerror("Error", "Please enter student name")
        return

    try:
        # Launch the data recording script with the name as argument
        subprocess.Popen(
            [sys.executable, RECORD_SCRIPT_PATH, name],
            cwd=BASE_DIR
        )
        messagebox.showinfo("Started", "Camera started for Data Recording.\nCapture will stop automatically after 30 images or click window and press Q to stop early.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def start_training():
    """Runs the script to train the recognition model."""
    try:
        # Launch the model trainer script
        subprocess.Popen(
            [sys.executable, TRAINER_SCRIPT_PATH],
            cwd=BASE_DIR
        )
        messagebox.showinfo("Training Started", "Model training started in a new process.\nCheck the console for progress. You MUST run this after recording new data.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def start_recognition_mode():
    """Starts the real-time face recognition and attendance taking."""
    try:
        # Launch the recognition script
        subprocess.Popen(
            [sys.executable, RECOGNITION_SCRIPT_PATH],
            cwd=BASE_DIR
        )
        messagebox.showinfo("Recognition Started", "Recognition mode started.\nClick on the camera window and press Q to stop.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def exit_app():
    root.destroy()

# ================= TKINTER GUI SETUP =================
root = tk.Tk()
root.title("Student Attendance System (Face Recognition)")
root.geometry("450x380")
root.resizable(False, False)

title = tk.Label(root, text="Face Recognition Attendance System",
                 font=("Arial", 16, "bold"))
title.pack(pady=20)

# --- Student Name Entry (for Data Recording) ---
label = tk.Label(root, text="1. New Student Name (for Recording):")
label.pack()

entry = tk.Entry(root, width=30)
entry.pack(pady=5)

# --- Buttons ---

# 1. Start Recording Button
start_btn = tk.Button(root, text="1. Start Face Data Recording",
                      width=30, height=2, bg="#3498db", fg="white",
                      command=start_attendance_recording)
start_btn.pack(pady=10)

# 2. Train Model Button (Needed after collecting new data)
train_btn = tk.Button(root, text="2. Train Recognition Model",
                      width=30, height=2, bg="#e67e22", fg="white",
                      command=start_training)
train_btn.pack(pady=10)

# 3. Start Recognition Button
recognize_btn = tk.Button(root, text="3. Start Recognition Mode (Attendance)",
                      width=30, height=2, bg="#27ae60", fg="white",
                      command=start_recognition_mode)
recognize_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit",
                     width=30, height=1, bg="#c0392b", fg="white",
                     command=exit_app)
exit_btn.pack(pady=10)

root.mainloop()
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "face_data_recorder.py")

def start_attendance():
    name = entry.get().strip()

    if not name:
        messagebox.showerror("Error", "Please enter student name")
        return

    try:
        subprocess.Popen(
            [sys.executable, SCRIPT_PATH, name],
            cwd=BASE_DIR
        )
        messagebox.showinfo("Started", "Camera started.\nPress Q to stop.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Student Attendance System")
root.geometry("400x250")
root.resizable(False, False)

title = tk.Label(root, text="Student Attendance System",
                 font=("Arial", 16, "bold"))
title.pack(pady=20)

label = tk.Label(root, text="Student Name:")
label.pack()

entry = tk.Entry(root, width=30)
entry.pack(pady=5)

start_btn = tk.Button(root, text="Start Attendance",
                      width=20, height=2,
                      command=start_attendance)
start_btn.pack(pady=15)

exit_btn = tk.Button(root, text="Exit",
                     width=20, command=exit_app)
exit_btn.pack()

root.mainloop()

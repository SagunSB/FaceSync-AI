import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk


DB_PATH = Path("database/attendance.db")
EXPORT_PATH = Path("attendance/attendance_export.xlsx")
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
FRAME_RESIZE_SCALE = 0.25
RECOGNITION_EVERY_N_FRAMES = 3
CPU_THREADS = os.cpu_count() or 4
MATCH_TOLERANCE = 0.6


class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("1000x650")
        self.root.minsize(900, 600)

        cv2.setUseOptimized(True)
        cv2.setNumThreads(CPU_THREADS)

        self.camera = None
        self.mode = None
        self.running = False
        self.frame_count = 0
        self.register_stable_frames = 0
        self.register_frames_needed = 30
        self.recognized_faces = []
        self.known_students = []
        self.known_encodings = []
        self.marked_today = set()

        self.student_code_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.stats_var = tk.StringVar(value="Students: 0   Attendance today: 0")

        self._build_style()
        self._build_layout()
        self.refresh_stats()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_style(self):
        self.root.configure(bg="#f4f7fb")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Sidebar.TFrame", background="#152238")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("App.TFrame", background="#f4f7fb")
        style.configure("Title.TLabel", background="#152238", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("Nav.TButton", font=("Segoe UI", 11), padding=(14, 10))
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 10))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 10))
        style.configure("Heading.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 16, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#e8f1ff", foreground="#12315f", font=("Segoe UI", 11, "bold"))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=240)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        title = ttk.Label(sidebar, text="Attendance\nSystem", style="Title.TLabel", justify="left")
        title.pack(anchor="w", padx=22, pady=(24, 28))

        ttk.Button(sidebar, text="Take Attendance", style="Nav.TButton", command=self.start_attendance_mode).pack(
            fill="x", padx=18, pady=6
        )
        ttk.Button(sidebar, text="Stop Camera", style="Nav.TButton", command=self.stop_camera).pack(
            fill="x", padx=18, pady=6
        )
        ttk.Button(sidebar, text="Export Excel", style="Nav.TButton", command=self.export_attendance).pack(
            fill="x", padx=18, pady=6
        )
        ttk.Button(sidebar, text="Refresh Tables", style="Nav.TButton", command=self.refresh_stats).pack(
            fill="x", padx=18, pady=6
        )

        self.stats_label = tk.Label(
            sidebar,
            textvariable=self.stats_var,
            bg="#152238",
            fg="#cbd5e1",
            font=("Segoe UI", 10),
            justify="left",
            wraplength=190,
        )
        self.stats_label.pack(anchor="w", padx=22, pady=(28, 0))

        content = ttk.Frame(self.root, style="App.TFrame")
        content.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(1, weight=1)

        header = ttk.Frame(content, style="Panel.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Live Face Attendance", style="Heading.TLabel").grid(row=0, column=0, sticky="w", padx=18, pady=(14, 2))
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=1, sticky="e", padx=18, pady=14)
        ttk.Label(
            header,
            text="Register students once, then use attendance mode to mark one record per student per day.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 14))

        video_panel = ttk.Frame(content, style="Panel.TFrame")
        video_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        video_panel.columnconfigure(0, weight=1)
        video_panel.rowconfigure(0, weight=1)

        self.video_label = tk.Label(video_panel, bg="#0f172a", fg="#64748b", text="Camera feed will appear here", font=("Segoe UI", 12))
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        side_panel = ttk.Frame(content, style="Panel.TFrame")
        side_panel.grid(row=1, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(6, weight=1)

        ttk.Label(side_panel, text="Student Registration", style="Heading.TLabel").grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))
        form = ttk.Frame(side_panel, style="Panel.TFrame")
        form.grid(row=1, column=0, sticky="ew", padx=16)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Student Code", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.student_code_var).grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=6)
        ttk.Label(form, text="Name", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.name_var).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=6)

        ttk.Button(side_panel, text="Start Registration", style="Primary.TButton", command=self.start_register_mode).grid(
            row=2, column=0, sticky="ew", padx=16, pady=(14, 6)
        )

        ttk.Label(side_panel, text="Student Management", style="Heading.TLabel").grid(row=3, column=0, sticky="w", padx=16, pady=(18, 8))
        ttk.Button(side_panel, text="Delete Student", style="Primary.TButton", command=self.delete_student).grid(
            row=4, column=0, sticky="ew", padx=16, pady=(0, 6)
        )

        ttk.Label(side_panel, text="Today's Attendance", style="Heading.TLabel").grid(row=5, column=0, sticky="w", padx=16, pady=(18, 8))
        self.attendance_tree = ttk.Treeview(side_panel, columns=("code", "name", "time"), show="headings", selectmode="browse")
        self.attendance_tree.heading("code", text="Code")
        self.attendance_tree.heading("name", text="Name")
        self.attendance_tree.heading("time", text="Time")
        self.attendance_tree.column("code", width=105, anchor="w")
        self.attendance_tree.column("name", width=145, anchor="w")
        self.attendance_tree.column("time", width=80, anchor="center")
        self.attendance_tree.grid(row=6, column=0, sticky="nsew", padx=16, pady=(0, 6))

        ttk.Button(side_panel, text="Delete Selected", style="Primary.TButton", command=self.delete_selected_attendance).grid(
            row=7, column=0, sticky="ew", padx=16, pady=(0, 16)
        )

    def open_camera(self):
        if self.camera is not None:
            return True

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = None
            self.set_status("Could not open webcam")
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return False

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def start_register_mode(self):
        code = self.student_code_var.get().strip()
        name = self.name_var.get().strip()

        if not code or not name:
            messagebox.showwarning("Missing Details", "Enter student code and name before registration.")
            return

        self.stop_camera(clear_image=False)
        self.mode = "register"
        self.running = True
        self.register_stable_frames = 0
        self.set_status("Registration mode: show one clear face")

        if self.open_camera():
            self.update_video()

    def start_attendance_mode(self):
        self.known_students = self.load_registered_students()
        if not self.known_students:
            messagebox.showwarning("No Students", "Register at least one student before taking attendance.")
            return

        self.known_encodings = [student["encoding"] for student in self.known_students]
        self.marked_today = self.load_marked_today()
        self.stop_camera(clear_image=False)
        self.mode = "attendance"
        self.running = True
        self.frame_count = 0
        self.recognized_faces = []
        self.set_status("Attendance mode: recognizing faces")

        if self.open_camera():
            self.update_video()

    def update_video(self):
        if not self.running or self.camera is None:
            return

        success, frame = self.camera.read()
        if not success:
            self.set_status("Could not read webcam frame")
            self.stop_camera()
            return

        if self.mode == "register":
            frame = self.process_registration_frame(frame)
        elif self.mode == "attendance":
            frame = self.process_attendance_frame(frame)

        self.show_frame(frame)
        self.root.after(10, self.update_video)

    def process_registration_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (30, 180, 90), 2)

        if len(face_locations) == 1:
            self.register_stable_frames += 1
            remaining = max(self.register_frames_needed - self.register_stable_frames, 0)
            message = f"Hold still. Saving in {remaining}"
        elif len(face_locations) == 0:
            self.register_stable_frames = 0
            message = "No face detected"
        else:
            self.register_stable_frames = 0
            message = "Only one face allowed"

        self.draw_banner(frame, message)

        if self.register_stable_frames >= self.register_frames_needed:
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if encodings:
                self.save_student(self.student_code_var.get().strip(), self.name_var.get().strip(), encodings[0])
            else:
                self.register_stable_frames = 0
                self.set_status("Encoding failed. Try again.")

        return frame

    def process_attendance_frame(self, frame):
        self.frame_count += 1
        scale_up = int(1 / FRAME_RESIZE_SCALE)

        if self.frame_count % RECOGNITION_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            self.recognized_faces = []

            for encoding, location in zip(face_encodings, face_locations):
                name = "Unknown"
                student = None
                distances = face_recognition.face_distance(self.known_encodings, encoding)

                if len(distances) > 0:
                    best_index = int(np.argmin(distances))
                    if distances[best_index] <= MATCH_TOLERANCE:
                        student = self.known_students[best_index]
                        name = student["name"]
                        self.mark_attendance(student)

                self.recognized_faces.append((name, location))

        for name, location in self.recognized_faces:
            top, right, bottom, left = location
            top *= scale_up
            right *= scale_up
            bottom *= scale_up
            left *= scale_up
            color = (30, 180, 90) if name != "Unknown" else (40, 80, 230)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, max(top - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        self.draw_banner(frame, "Attendance running")
        return frame

    def draw_banner(self, frame, text):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (15, 23, 42), -1)
        cv2.putText(frame, text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((640, 480), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def save_student(self, student_code, name, face_encoding):
        encoding_json = json.dumps(face_encoding.tolist())
        created_at = datetime.now().isoformat(timespec="seconds")

        try:
            with sqlite3.connect(DB_PATH) as connection:
                connection.execute(
                    """
                    INSERT INTO students (student_code, name, face_encoding, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (student_code, name, encoding_json, created_at),
                )
        except sqlite3.IntegrityError:
            self.register_stable_frames = 0
            self.set_status(f"Student code already exists: {student_code}")
            messagebox.showwarning("Duplicate Student", f"Student code '{student_code}' already exists.")
            return

        self.set_status(f"Registered {name}")
        messagebox.showinfo("Registered", f"{name} registered successfully.")
        self.student_code_var.set("")
        self.name_var.set("")
        self.refresh_stats()
        self.stop_camera()

    def load_registered_students(self):
        with sqlite3.connect(DB_PATH) as connection:
            rows = connection.execute(
                """
                SELECT id, student_code, name, face_encoding
                FROM students
                ORDER BY id
                """
            ).fetchall()

        students = []
        for student_id, code, name, encoding_json in rows:
            students.append(
                {
                    "id": student_id,
                    "student_code": code,
                    "name": name,
                    "encoding": np.array(json.loads(encoding_json)),
                }
            )
        return students

    def mark_attendance(self, student):
        today = date.today().isoformat()
        if student["id"] in self.marked_today:
            return

        now = datetime.now()
        attendance_time = now.strftime("%H:%M:%S")
        created_at = now.isoformat(timespec="seconds")

        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO attendance
                    (student_id, attendance_date, attendance_time, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (student["id"], today, attendance_time, created_at),
            )

        self.marked_today.add(student["id"])
        if cursor.rowcount:
            self.set_status(f"Marked {student['name']} at {attendance_time}")
            self.refresh_stats()

    def load_marked_today(self):
        today = date.today().isoformat()
        with sqlite3.connect(DB_PATH) as connection:
            rows = connection.execute(
                "SELECT student_id FROM attendance WHERE attendance_date = ?",
                (today,),
            ).fetchall()
        return {row[0] for row in rows}

    def refresh_stats(self):
        today = date.today().isoformat()
        with sqlite3.connect(DB_PATH) as connection:
            student_count = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            attendance_count = connection.execute(
                "SELECT COUNT(*) FROM attendance WHERE attendance_date = ?",
                (today,),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT s.student_code, s.name, a.attendance_time
                FROM attendance a
                JOIN students s ON s.id = a.student_id
                WHERE a.attendance_date = ?
                ORDER BY a.attendance_time DESC
                """,
                (today,),
            ).fetchall()

        self.stats_var.set(f"Students: {student_count}\nAttendance today: {attendance_count}")
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        for code, name, attendance_time in rows:
            self.attendance_tree.insert("", "end", values=(code, name, attendance_time))

    def export_attendance(self):
        EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as connection:
            query = """
                SELECT
                    s.student_code AS student_code,
                    s.name AS name,
                    a.attendance_date AS attendance_date,
                    a.attendance_time AS attendance_time,
                    a.created_at AS created_at
                FROM attendance a
                JOIN students s ON s.id = a.student_id
                ORDER BY a.attendance_date DESC, a.attendance_time DESC
            """
            data = pd.read_sql_query(query, connection)

        data.to_excel(EXPORT_PATH, index=False)
        self.set_status(f"Exported {EXPORT_PATH}")
        messagebox.showinfo("Export Complete", f"Attendance exported to:\n{EXPORT_PATH}")

    def delete_student(self):
        students = self.load_registered_students()
        if not students:
            messagebox.showwarning("No Students", "No registered students to delete.")
            return

        # Create a dialog to select student
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete Student")
        dialog.geometry("400x300")
        dialog.resizable(False, False)

        ttk.Label(dialog, text="Select student to delete:").pack(pady=10)

        student_var = tk.StringVar()
        student_combo = ttk.Combobox(dialog, textvariable=student_var, state="readonly")
        student_combo["values"] = [f"{s['student_code']} - {s['name']}" for s in students]
        student_combo.pack(pady=5, padx=20, fill="x")

        def confirm_delete():
            selected = student_var.get()
            if not selected:
                messagebox.showwarning("No Selection", "Please select a student to delete.")
                return

            student_code = selected.split(" - ")[0]
            student = next((s for s in students if s["student_code"] == student_code), None)
            if not student:
                return

            if not messagebox.askyesno("Confirm Delete", f"Delete student {student['name']} ({student_code})?\nThis will also delete all their attendance records."):
                return

            # Delete attendance records first
            with sqlite3.connect(DB_PATH) as connection:
                connection.execute("DELETE FROM attendance WHERE student_id = ?", (student["id"],))
                connection.execute("DELETE FROM students WHERE id = ?", (student["id"],))

            self.set_status(f"Deleted student {student['name']}")
            self.refresh_stats()
            dialog.destroy()

        ttk.Button(dialog, text="Delete", command=confirm_delete).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()

    def delete_selected_attendance(self):
        selected_item = self.attendance_tree.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select an attendance record to delete.")
            return

        item = self.attendance_tree.item(selected_item)
        values = item["values"]
        if not values or len(values) != 3:
            return

        student_code, name, time = values
        today = date.today().isoformat()

        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", f"Delete attendance for {name} ({student_code}) at {time}?"):
            return

        # Delete from database
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute(
                """
                DELETE FROM attendance
                WHERE attendance_date = ? AND attendance_time = ? AND student_id = (
                    SELECT id FROM students WHERE student_code = ?
                )
                """,
                (today, time, student_code),
            )

        if cursor.rowcount > 0:
            self.set_status(f"Deleted attendance for {name}")
            # Remove from marked_today if it was today's attendance
            with sqlite3.connect(DB_PATH) as connection:
                student_id = connection.execute(
                    "SELECT id FROM students WHERE student_code = ?",
                    (student_code,),
                ).fetchone()
                if student_id:
                    self.marked_today.discard(student_id[0])
            self.refresh_stats()
        else:
            messagebox.showwarning("Delete Failed", "Could not delete the attendance record.")

    def set_status(self, message):
        self.status_var.set(message)

    def stop_camera(self, clear_image=True):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if clear_image:
            self.video_label.configure(image="", text="Camera feed will appear here")
            self.video_label.image = None
        if self.mode:
            self.set_status("Camera stopped")
        self.mode = None

    def close(self):
        self.stop_camera(clear_image=False)
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

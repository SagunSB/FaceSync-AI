import json
import os
import sqlite3
from datetime import datetime

import cv2
import face_recognition
import numpy as np


DB_PATH = "database/attendance.db"
FRAME_RESIZE_SCALE = 0.25
RECOGNITION_EVERY_N_FRAMES = 3
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CPU_THREADS = os.cpu_count() or 4


def save_student(student_code, name, face_encoding):
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
        print(f"Error: Student code '{student_code}' already exists.")
        return False

    return True


def load_registered_students():
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT id, student_code, name, face_encoding
            FROM students
            ORDER BY id
            """
        ).fetchall()

    students = []

    for student_id, student_code, name, encoding_json in rows:
        encoding = np.array(json.loads(encoding_json))
        students.append(
            {
                "id": student_id,
                "student_code": student_code,
                "name": name,
                "encoding": encoding,
            }
        )

    return students


def recognize_students():
    students = load_registered_students()

    if not students:
        print("No registered students found. Register a student first.")
        return

    known_encodings = [student["encoding"] for student in students]
    known_names = [student["name"] for student in students]

    cv2.setUseOptimized(True)
    cv2.setNumThreads(CPU_THREADS)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Recognition started. Press 'q' in the webcam window to exit.")
    frame_count = 0
    recognized_faces = []
    scale_up = int(1 / FRAME_RESIZE_SCALE)

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame from webcam.")
            break

        frame_count += 1

        if frame_count % RECOGNITION_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(
                frame,
                (0, 0),
                fx=FRAME_RESIZE_SCALE,
                fy=FRAME_RESIZE_SCALE,
            )
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                model="hog",
            )
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame,
                face_locations,
            )
            recognized_faces = []

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    known_encodings,
                    face_encoding,
                    tolerance=0.6,
                )
                name = "Unknown"

                face_distances = face_recognition.face_distance(
                    known_encodings,
                    face_encoding,
                )

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                recognized_faces.append((name, face_location))

        for name, face_location in recognized_faces:
            top, right, bottom, left = face_location
            top *= scale_up
            right *= scale_up
            bottom *= scale_up
            left *= scale_up

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Attendance System - Recognize Student", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def register_student():
    student_code = input("Enter student code: ").strip()
    name = input("Enter student name: ").strip()

    if not student_code or not name:
        print("Error: Student code and name are required.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Look at the camera. The system will auto-save after a stable face is detected.")
    print("Press 'q' in the webcam window to cancel.")
    status_message = "Show one clear face to auto-save"
    stable_face_frames = 0
    frames_needed_to_save = 30

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if len(face_locations) == 1:
            stable_face_frames += 1
            remaining = max(frames_needed_to_save - stable_face_frames, 0)
            status_message = f"Hold still... saving in {remaining}"
        elif len(face_locations) == 0:
            stable_face_frames = 0
            status_message = "No face detected. Improve lighting."
        else:
            stable_face_frames = 0
            status_message = "Multiple faces detected. Show only one face."

        cv2.putText(
            frame,
            status_message,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Face Attendance System - Register Student", frame)

        key = cv2.waitKey(1) & 0xFF

        if stable_face_frames >= frames_needed_to_save:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if not face_encodings:
                status_message = "Could not create face encoding. Try again."
                print(status_message)
                stable_face_frames = 0
                continue

            saved = save_student(student_code, name, face_encodings[0])

            if saved:
                status_message = f"Registered {name} successfully."
                print(status_message)
                cv2.putText(
                    frame,
                    status_message,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Face Attendance System - Register Student", frame)
                cv2.waitKey(1200)
                break

            stable_face_frames = 0

        if key == ord("q"):
            print("Registration cancelled.")
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    print("Face Attendance System")
    print("1. Register student")
    print("2. Recognize student")

    choice = input("Choose an option: ").strip()

    if choice == "1":
        register_student()
    elif choice == "2":
        recognize_students()
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    face_encoding TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    attendance_date TEXT NOT NULL,
    attendance_time TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id),
    UNIQUE(student_id, attendance_date)
);

CREATE INDEX IF NOT EXISTS idx_students_student_code
ON students(student_code);

CREATE INDEX IF NOT EXISTS idx_attendance_date
ON attendance(attendance_date);

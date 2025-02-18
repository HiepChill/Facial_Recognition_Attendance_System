from flask import Blueprint, request, jsonify
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime

attendance_bp = Blueprint('attendance', __name__)

DATABASE = "backend/database/attendance.db"

# Nhận diện và ghi log điểm danh
@attendance_bp.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["image"]
    image = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return jsonify({"error": "Không tìm thấy khuôn mặt"}), 400

    face_encoding = encodings[0]

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encoding FROM employees")
    employees = cursor.fetchall()

    for emp in employees:
        emp_id, name, stored_encoding = emp
        known_encoding = np.frombuffer(stored_encoding, dtype=np.float64)

        match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
        if match:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO attendance_logs (employee_id, timestamp) VALUES (?, ?)", (emp_id, timestamp))
            conn.commit()
            conn.close()
            return jsonify({"message": f"{name} đã điểm danh vào lúc {timestamp}"})

    conn.close()
    return jsonify({"message": "Không nhận diện được nhân viên"}), 404

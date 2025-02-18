from flask import Blueprint, request, jsonify
import face_recognition
import numpy as np
import sqlite3

employees_bp = Blueprint('employees', __name__)

DATABASE = "backend/database/attendance.db"

# Thêm nhân viên mới
@employees_bp.route("/add", methods=["POST"])
def add_employee():
    data = request.json
    name = data["name"]
    image_path = f"dataset/{name}.jpg"

    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if not encoding:
        return jsonify({"error": "Không nhận diện được khuôn mặt"}), 400

    encoding_bytes = encoding[0].tobytes()
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO employees (name, encoding) VALUES (?, ?)", (name, encoding_bytes))
    conn.commit()
    conn.close()

    return jsonify({"message": f"Đã thêm {name} vào danh sách"})


# Lấy danh sách nhân viên
@employees_bp.route("/list", methods=["GET"])
def get_employees():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM employees")
    employees = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    conn.close()
    return jsonify({"employees": employees})

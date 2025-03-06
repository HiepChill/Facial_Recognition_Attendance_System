import sqlite3
import os
from .config import DB_PATH
from datetime import datetime

def setup_database():
    """Thiết lập cơ sở dữ liệu"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tạo bảng users với id dạng UUID string
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Tạo bảng face_images
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Thêm bảng để lưu trạng thái điểm danh gần nhất
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance_status (
        user_id TEXT PRIMARY KEY,
        last_event TEXT NOT NULL,
        last_time TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def check_user_exists(user_id):
    """Kiểm tra user_id đã tồn tại chưa"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def add_user(user_id, name):
    """Thêm người dùng mới"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        (user_id, name)
    )
    conn.commit()
    conn.close()

def add_face_image(user_id, image_path):
    """Lưu đường dẫn ảnh vào database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO face_images (user_id, image_path) VALUES (?, ?)",
        (user_id, image_path)
    )
    conn.commit()
    conn.close()

def get_all_users():
    """Lấy danh sách tất cả người dùng"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM users")
    users = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    conn.close()
    return users

def get_user_face_images(user_id):
    """Lấy danh sách ảnh khuôn mặt của người dùng"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM face_images WHERE user_id = ?", (user_id,))
    face_images = [row[0] for row in cursor.fetchall()]
    conn.close()
    return face_images

def get_user_face_data():
    """Lấy thông tin người dùng và đường dẫn ảnh khuôn mặt"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT u.id, u.name, f.image_path 
    FROM users u
    JOIN face_images f ON u.id = f.user_id
    ''')
    results = cursor.fetchall()
    conn.close()
    return results

def get_last_attendance_status(user_id):
    """Lấy trạng thái điểm danh gần nhất của người dùng"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT last_event, last_time FROM attendance_status WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result  # Trả về (last_event, last_time) hoặc None nếu chưa có

def update_attendance_status(user_id, event, timestamp):
    """Cập nhật trạng thái điểm danh gần nhất"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO attendance_status (user_id, last_event, last_time)
    VALUES (?, ?, ?)
    """, (user_id, event, timestamp))
    conn.commit()
    conn.close()
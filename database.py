import sqlite3
import os
from datetime import datetime

DB_PATH = "attendance.db"

def get_db_connection():
    """Tạo kết nối tới cơ sở dữ liệu SQLite"""
    conn = sqlite3.connect(DB_PATH)
    return conn

def setup_database():
    """Thiết lập cơ sở dữ liệu với các bảng cần thiết"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Thêm bảng mới để lưu trạng thái điểm danh gần nhất
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

def load_user_faces():
    """Tải thông tin người dùng và ảnh khuôn mặt từ database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT u.id, u.name, f.image_path 
    FROM users u
    JOIN face_images f ON u.id = f.user_id
    ''')
    
    results = cursor.fetchall()
    conn.close()
    return results

def insert_user(user_id: str, name: str):
    """Thêm người dùng mới vào database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (id, name) VALUES (?, ?)", (user_id, name))
    conn.commit()
    conn.close()

def insert_face_image(user_id: str, image_path: str):
    """Thêm đường dẫn ảnh khuôn mặt vào database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO face_images (user_id, image_path) VALUES (?, ?)", (user_id, image_path))
    conn.commit()
    conn.close()

def get_users():
    """Lấy danh sách tất cả người dùng"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM users")
    users = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    conn.close()
    return users

def get_user_faces(user_id: str):
    """Lấy danh sách ảnh khuôn mặt của người dùng"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM face_images WHERE user_id = ?", (user_id,))
    face_images = [row[0] for row in cursor.fetchall()]
    conn.close()
    return face_images

def get_last_attendance_status(user_id: str):
    """Lấy trạng thái điểm danh gần nhất của người dùng"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT last_event, last_time FROM attendance_status WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result  # Trả về (last_event, last_time) hoặc None nếu chưa có

def update_attendance_status(user_id: str, event: str, timestamp: datetime):
    """Cập nhật trạng thái điểm danh gần nhất"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO attendance_status (user_id, last_event, last_time)
    VALUES (?, ?, ?)
    """, (user_id, event, timestamp))
    conn.commit()
    conn.close()
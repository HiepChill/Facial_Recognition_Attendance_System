import cv2
import os
import numpy as np
import insightface
import sqlite3
import time
import csv
import uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from pydantic import BaseModel
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Khởi tạo FastAPI app
app = FastAPI(title="Face Recognition Attendance System")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình đường dẫn
DB_PATH = "attendance.db"
DATASET_DIR = "dataset"
ATTENDANCE_DIR = "attendance_logs"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Khởi tạo mô hình nhận diện khuôn mặt
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Từ điển lưu thời gian điểm danh gần nhất của mỗi người
last_attendance_time = {}

# Từ điển lưu embeddings của tất cả người dùng đã đăng ký
face_database = {}

# Models Pydantic
class User(BaseModel):
    id: Optional[str] = None
    name: str

class AttendanceRecord(BaseModel):
    name: str
    user_id: str
    time: str
    event: str

# if os.path.exists(DB_PATH):
#     os.remove(DB_PATH)

# Thiết lập cơ sở dữ liệu
def setup_database():
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
    
    conn.commit()
    conn.close()

# Gọi setup_database khi ứng dụng khởi động
setup_database()

# Load face database từ SQLite
def load_face_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Lấy thông tin người dùng và đường dẫn ảnh
    cursor.execute('''
    SELECT u.id, u.name, f.image_path 
    FROM users u
    JOIN face_images f ON u.id = f.user_id
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    face_db = {}
    user_info = {}
    
    for user_id, name, image_path in results:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            faces = face_analyzer.get(img)
            if faces:
                embedding = faces[0].normed_embedding
                user_key = f"{user_id}_{name}"
                
                if user_key not in face_db:
                    face_db[user_key] = []
                    user_info[user_key] = {"name": name, "user_id": user_id}
                
                face_db[user_key].append(embedding)
    
    # Tính trung bình các embedding cho mỗi người dùng (nếu có nhiều ảnh)
    averaged_db = {}
    for user_key, embeddings in face_db.items():
        if embeddings:
            averaged_embedding = np.mean(embeddings, axis=0)
            averaged_db[user_key] = {
                "embedding": averaged_embedding,
                "info": user_info[user_key]
            }
    
    return averaged_db

# Hàm ghi nhận điểm danh
def log_attendance(name, user_id, event_type):
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    
    # Tạo file CSV với header nếu chưa tồn tại
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'User ID', 'Time', 'Event'])
    
    # Ghi thông tin điểm danh
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, user_id, current_time, event_type])

# Kiểm tra thời gian giữa 2 lần điểm danh (ít nhất 10 phút)
def can_record_attendance(user_id):
    current_time = datetime.now()
    
    if user_id in last_attendance_time:
        last_time = last_attendance_time[user_id]
        time_diff = current_time - last_time
        
        # Nếu chưa đủ 10 phút, không ghi nhận
        if time_diff < timedelta(minutes=10):
            return False
    
    # Cập nhật thời gian điểm danh gần nhất
    last_attendance_time[user_id] = current_time
    return True

# Load dữ liệu khuôn mặt khi khởi động
face_database = load_face_database()

# API endpoints
@app.post("/register_face")
async def register_face(
    name: str = Form(...),
    face_images: List[UploadFile] = File(...)
):
    """Đăng ký khuôn mặt và tạo người dùng mới với nhiều ảnh"""
    if not face_images:
        return JSONResponse(status_code=400, content={"error": "No images uploaded"})
    
    # Tạo ID duy nhất cho người dùng
    user_id = str(uuid.uuid4())
    
    # Kết nối cơ sở dữ liệu
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Thêm người dùng mới
        cursor.execute(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            (user_id, name)
        )
        
        # Tạo thư mục người dùng
        user_folder = os.path.join(DATASET_DIR, f"{user_id}_{name}")
        os.makedirs(user_folder, exist_ok=True)
        
        saved_images = []
        
        # Xử lý từng ảnh
        for face_image in face_images:
            contents = await face_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
            
            # Phát hiện khuôn mặt
            faces = face_analyzer.get(img)
            
            if not faces:
                continue
            
            if len(faces) > 1:
                continue  # Skip images with multiple faces
            
            # Lưu ảnh
            image_filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join(user_folder, image_filename)
            cv2.imwrite(image_path, img)
            
            # Lưu đường dẫn vào database
            cursor.execute(
                "INSERT INTO face_images (user_id, image_path) VALUES (?, ?)",
                (user_id, image_path)
            )
            
            saved_images.append(image_path)
        
        # Nếu không có ảnh nào hợp lệ, rollback và trả về lỗi
        if not saved_images:
            conn.rollback()
            return JSONResponse(status_code=400, content={"error": "No valid face images found"})
        
        conn.commit()
        
        # Cập nhật face database
        global face_database
        face_database = load_face_database()
        
        return {
            "message": "User registered successfully", 
            "user_id": user_id,
            "name": name,
            "image_count": len(saved_images)
        }
        
    except Exception as e:
        conn.rollback()
        return JSONResponse(status_code=500, content={"error": f"Registration failed: {str(e)}"})
    finally:
        conn.close()

@app.get("/users")
async def get_users():
    """Lấy danh sách tất cả người dùng"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name FROM users")
    users = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return {"users": users}

@app.get("/user/{user_id}/faces")
async def get_user_faces(user_id: str):
    """Lấy danh sách ảnh khuôn mặt của người dùng"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_path FROM face_images WHERE user_id = ?", (user_id,))
    face_images = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return {"user_id": user_id, "face_images": face_images}

@app.get("/attendance/{date}")
async def get_attendance(date: str):
    """Lấy dữ liệu điểm danh theo ngày (định dạng: YYYY-MM-DD)"""
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date}.csv")
    
    if not os.path.exists(csv_path):
        return {"date": date, "records": []}
    
    records = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Bỏ qua header
        
        for row in reader:
            if len(row) >= 4:
                records.append({
                    "name": row[0],
                    "user_id": row[1],
                    "time": row[2],
                    "event": row[3]
                })
    
    return {"date": date, "records": records}

@app.get("/today_attendance")
async def get_today_attendance():
    """Lấy dữ liệu điểm danh của ngày hôm nay"""
    today = datetime.now().strftime("%Y-%m-%d")
    return await get_attendance(today)

# Xử lý frame để nhận diện khuôn mặt
# Xử lý frame để nhận diện khuôn mặt
def process_frame(frame):
    # Copy frame để vẽ lên
    display_frame = frame.copy()
    recognized_users = []
    
    try:
        # Phát hiện khuôn mặt trong frame
        faces = face_analyzer.get(frame)
        
        for face in faces:
            # Lấy embedding của khuôn mặt
            face_embedding = face.normed_embedding
            
            # Biến lưu thông tin người được nhận diện
            match_info = None
            max_similarity = -1  # Bắt đầu với giá trị âm để đảm bảo luôn tìm được giá trị tương đồng cao nhất
            
            # So sánh với database
            for user_key, data in face_database.items():
                db_embedding = data["embedding"]
                similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    if similarity > 0.5:  # Ngưỡng nhận diện
                        match_info = data["info"]
                    else:
                        match_info = None
            
            # Lấy tọa độ khuôn mặt
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            
            if match_info:
                # Vẽ khung xanh nếu nhận diện được
                color = (0, 255, 0)
                name = match_info["name"]
                user_id = match_info["user_id"]
                
                # Hiển thị tên và độ tương đồng
                label = f"{name} ({max_similarity:.2f})"
                
                # Kiểm tra và ghi nhận điểm danh
                if can_record_attendance(user_id):
                    event_type = "check-in"
                    log_attendance(name, user_id, event_type)
                    
                    recognized_users.append({
                        "user_id": user_id,
                        "name": name,
                        "event": event_type,
                        "confidence": float(max_similarity)
                    })
            else:
                # Vẽ khung đỏ nếu không nhận diện được
                color = (0, 0, 255)
                label = f"Unknown ({max_similarity:.2f})"
            
            # Vẽ bounding box
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Hiển thị nhãn
            y_position = max(top - 10, 20)  # Đảm bảo text không bị cắt khỏi frame
            cv2.putText(display_frame, 
                        label,
                        (left, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
    
    except Exception as e:
        print(f"Error in face recognition: {e}")
    
    return display_frame, recognized_users

# Generator để stream video
def generate_frames():
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Xử lý frame cho nhận diện
            processed_frame, _ = process_frame(frame)
            
            # Chuyển đổi frame thành JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame cho streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Giảm tốc độ frame
            time.sleep(0.05)
    
    finally:
        cap.release()

# Stream video với nhận diện khuôn mặt
@app.get("/video_feed")
async def video_feed():
    """Stream video với nhận diện khuôn mặt"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("startup")
async def startup_event():
    """Tải face database khi khởi động ứng dụng"""
    global face_database
    face_database = load_face_database()
    print(f"Loaded {len(face_database)} users into face database")

@app.get("/")
async def root():
    return {"message": "Face Recognition Attendance System"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
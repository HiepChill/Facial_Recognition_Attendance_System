from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import cv2
import numpy as np
import os
from datetime import datetime
import uvicorn

from database import setup_database, load_user_faces, insert_user, insert_face_image, get_users, get_user_faces
from face_recognition import load_face_database, process_frame
from utils import log_attendance, get_attendance

# Khởi tạo FastAPI app
app = FastAPI(title="Face Recognition Attendance System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cần giới hạn trong sản xuất
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình
DATASET_DIR = "dataset"
face_database = {}

@app.on_event("startup")
async def startup_event():
    """Khởi tạo ứng dụng"""
    setup_database()
    global face_database
    user_data = load_user_faces()
    face_database = load_face_database(user_data)
    print(f"Loaded {len(face_database)} users into face database")

@app.post("/register_face")
async def register_face(
    user_id: str = Form(...),
    name: str = Form(...),
    face_images: List[UploadFile] = File(...)
):
    """Đăng ký khuôn mặt mới"""
    if not face_images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    
    insert_user(user_id, name)
    user_folder = os.path.join(DATASET_DIR, f"{user_id}_{name}")
    os.makedirs(user_folder, exist_ok=True)
    
    saved_images = []
    sequence_number = 1
    
    for face_image in face_images:
        contents = await face_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        image_filename = f"{name}_{sequence_number:04d}.jpg"
        sequence_number += 1
        image_path = os.path.join(user_folder, image_filename)
        cv2.imwrite(image_path, img)
        insert_face_image(user_id, image_path)
        saved_images.append(image_path)
    
    if not saved_images:
        raise HTTPException(status_code=400, detail="No valid face images found")
    
    global face_database
    user_data = load_user_faces()
    face_database = load_face_database(user_data)
    
    return {
        "message": "User registered successfully",
        "user_id": user_id,
        "name": name,
        "image_count": len(saved_images)
    }

def generate_frames():
    """Stream video với nhận diện khuôn mặt"""
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            processed_frame, recognized_users = process_frame(frame, face_database)
            for user in recognized_users:
                log_attendance(user["name"], user["user_id"], user["event"])
            
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.get("/video_feed")
async def video_feed():
    """API stream video"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/users")
async def list_users():
    """Lấy danh sách người dùng"""
    return {"users": get_users()}

@app.get("/user/{user_id}/faces")
async def list_user_faces(user_id: str):
    """Lấy danh sách ảnh của người dùng"""
    return {"user_id": user_id, "face_images": get_user_faces(user_id)}

@app.get("/attendance/{date}")
async def get_attendance_by_date(date: str):
    """Lấy dữ liệu điểm danh theo ngày"""
    return {"date": date, "records": get_attendance(date)}

@app.get("/today_attendance")
async def get_today_attendance():
    """Lấy dữ liệu điểm danh hôm nay"""
    today = datetime.now().strftime("%Y-%m-%d")
    return {"date": today, "records": get_attendance(today)}

@app.get("/")
async def root():
    return {"message": "Face Recognition Attendance System"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
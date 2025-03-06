import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from contextlib import asynccontextmanager
from datetime import datetime
import os
import numpy as np

from app.config import DATASET_DIR
from app.database import setup_database, check_user_exists, add_user, add_face_image, get_all_users, get_user_face_images, get_user_face_data
from app.face_recognition import face_analyzer, face_database, load_face_database, process_frame
from app.camera import CameraManager
from app.attendance import get_attendance_records

# Tạo context manager để tải face database khi khởi động ứng dụng
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Tải face database khi khởi động ứng dụng và giải phóng tài nguyên khi đóng"""
    global face_database
    setup_database()
    user_face_data = get_user_face_data()
    face_database = load_face_database(user_face_data)
    print(f"Đã tải {len(face_database)} người dùng vào CSDL")
    
    yield  # Ứng dụng hoạt động

    # Giải phóng tài nguyên camera khi ứng dụng đóng
    camera_manager = CameraManager.get_instance()
    camera_manager.release_camera()
    print("Ứng dụng đang tắt: Đã giải phóng tài nguyên camera")

# Thêm CORS middleware: giúp kiểm soát quyền truy cập tài nguyên giữa các trang web có nguồn gốc khác nhau
app = FastAPI(title="Face Recognition Attendance System", lifespan=lifespan)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các nguồn (domain) truy cập API
        # allow_origins=["https://my-frontend.com", "https://admin.my-frontend.com"] # Chỉ cho phép các domain này truy cập API
    allow_credentials=True, # Cho phép gửi cookies, headers xác thực (hữu ích nếu API yêu cầu đăng nhập).
    allow_methods=["*"], # Cho phép tất cả phương thức HTTP (GET, POST, PUT, DELETE...)
    allow_headers=["*"], # Cho phép tất cả các headers được gửi trong request
)

# API endpoints
# Generator để stream video
def generate_frames():
    """Generator để stream video"""
    camera_manager = CameraManager.get_instance()
    cap = camera_manager.get_camera()
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Xử lý frame cho nhận diện
            processed_frame, _ = process_frame(frame, face_database)
            
            # Chuyển đổi frame thành JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame cho streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Lỗi stream: {e}")

# Stream video với nhận diện khuôn mặt
@app.get("/video_feed")
async def video_feed():
    """Stream video với nhận diện khuôn mặt"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/")
async def root():
    return {"message": "Face Recognition Attendance System"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
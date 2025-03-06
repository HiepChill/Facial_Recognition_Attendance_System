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

# API endpoints
@app.post("/register_face")
async def register_face(
    user_id: str = Form(...),  
    name: str = Form(...),
    face_images: List[UploadFile] = File(...)
):
    """Đăng ký khuôn mặt và tạo người dùng mới với nhiều ảnh"""
    if not face_images:
        return JSONResponse(status_code=400, content={"error": "Không có ảnh nào được chọn"})
    
    try:
        # Kiểm tra id đã tồn tại hay chưa
        if check_user_exists(user_id):
            return JSONResponse(
                status_code=400, 
                content={"error": f"ID người dùng '{user_id}' đã tồn tại. Vui lòng chọn ID khác."}
            )
        
        # Thêm người dùng mới
        add_user(user_id, name)
        
        # Tạo thư mục người dùng
        user_folder = os.path.join(DATASET_DIR, f"{user_id}_{name}")
        os.makedirs(user_folder, exist_ok=True)
        
        saved_images = []
        sequence_number = 1
        
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
                continue  # Bỏ qua những bức ảnh có nhiều hơn 1 khuôn mặt
            
            # Tạo tên file theo cấu trúc name_0001.jpg
            image_filename = f"{name}_{sequence_number:04d}.jpg"
            sequence_number += 1
            
            image_path = os.path.join(user_folder, image_filename)
            success = cv2.imwrite(image_path, img)
            if not success:
                print(f"Lỗi: Không thể lưu ảnh tại {image_path}")
                continue
            
            # Lưu đường dẫn vào database
            add_face_image(user_id, image_path)
            saved_images.append(image_path)
        
        # Nếu không có ảnh nào hợp lệ, trả về lỗi
        if not saved_images:
            return JSONResponse(status_code=400, content={"error": "Không có ảnh hợp lệ"})
        
        # Cập nhật face database
        global face_database
        user_face_data = get_user_face_data()
        face_database = load_face_database(user_face_data)
        
        return {
            "message": "Đăng ký thành công", 
            "user_id": user_id,
            "name": name,
            "image_count": len(saved_images)
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Đăng ký thất bại: {str(e)}"})

@app.get("/users")
async def get_users():
    """Lấy danh sách tất cả người dùng"""
    users = get_all_users()
    return {"users": users}

@app.get("/user/{user_id}/faces")
async def get_user_faces(user_id: str):
    """Lấy danh sách ảnh khuôn mặt của người dùng"""
    face_images = get_user_face_images(user_id)
    return {"user_id": user_id, "face_images": face_images}

@app.get("/attendance/{date}")
async def get_attendance(date: str):
    """Lấy dữ liệu điểm danh theo ngày (định dạng: YYYY-MM-DD)"""
    records = get_attendance_records(date)
    return {"date": date, "records": records}

@app.get("/today_attendance")
async def get_today_attendance():
    """Lấy dữ liệu điểm danh của ngày hôm nay"""
    records = get_attendance_records()
    today = datetime.now().strftime("%Y-%m-%d")
    return {"date": today, "records": records}
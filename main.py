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
app = FastAPI(title="Face Recognition Attendance System", lifespan=lifespan, description="API cho hệ thống điểm danh sử dụng nhận diện khuôn mặt. Cung cấp các chức năng đăng ký khuôn mặt, xem danh sách người dùng, và tra cứu dữ liệu điểm danh.")

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
@app.post("/register_face", 
    summary="Đăng ký khuôn mặt người dùng mới",
    description="API này nhận ID, tên người dùng và các ảnh khuôn mặt để đăng ký người dùng mới vào hệ thống. Các ảnh sẽ được xử lý để trích xuất đặc trưng khuôn mặt.",
    response_description="Thông tin người dùng đã được đăng ký và số lượng ảnh hợp lệ"
)

async def register_face(
    user_id: str = Form(..., description="Mã định danh của người dùng"),  
    name: str = Form(..., description="Họ tên của người dùng"),
    face_images: List[UploadFile] = File(..., description="Danh sách các ảnh khuôn mặt của người dùng")
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

@app.get("/users",
    summary="Lấy danh sách người dùng",
    description="Trả về danh sách tất cả người dùng đã đăng ký trong hệ thống.",
    response_description="Danh sách các người dùng với ID và tên"
)
async def get_users():
    """Lấy danh sách tất cả người dùng"""
    users = get_all_users()
    return {"users": users}

@app.get("/user/{user_id}/faces",
    summary="Lấy ảnh khuôn mặt của người dùng",
    description="Trả về danh sách đường dẫn đến các ảnh khuôn mặt của người dùng được chỉ định bởi user_id.",
    response_description="Danh sách đường dẫn đến các ảnh khuôn mặt"
)
async def get_user_faces(user_id: str):
    """Lấy danh sách ảnh khuôn mặt của người dùng"""
    face_images = get_user_face_images(user_id)
    return {"user_id": user_id, "face_images": face_images}

@app.get("/attendance/{date}",
    summary="Lấy dữ liệu điểm danh theo ngày",
    description="Trả về dữ liệu điểm danh của tất cả người dùng trong ngày được chỉ định, định dạng YYYY-MM-DD.",
    response_description="Danh sách bản ghi điểm danh trong ngày chỉ định"
)
async def get_attendance(date: str):
    """Lấy dữ liệu điểm danh theo ngày (định dạng: YYYY-MM-DD)"""
    records = get_attendance_records(date)
    return {"date": date, "records": records}

@app.get("/today_attendance",
    summary="Lấy dữ liệu điểm danh hôm nay",
    description="Trả về dữ liệu điểm danh của tất cả người dùng trong ngày hiện tại.",
    response_description="Danh sách bản ghi điểm danh trong ngày hiện tại"
)
async def get_today_attendance():
    """Lấy dữ liệu điểm danh của ngày hôm nay"""
    records = get_attendance_records()
    today = datetime.now().strftime("%Y-%m-%d")
    return {"date": today, "records": records}

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
@app.get("/video_feed",
    summary="Stream video có tích hợp nhận diện khuôn mặt",
    description="Cung cấp luồng video từ camera có tích hợp nhận diện khuôn mặt và điểm danh tự động. Stream có thể được hiển thị trên giao diện web.",
    response_description="Luồng video JPEG được phân đoạn",
    responses={
        200: {
            "content": {"multipart/x-mixed-replace": {}},
            "description": "Luồng video từ camera"
        }
    }
)
async def video_feed():
    """Stream video với nhận diện khuôn mặt"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/",
    summary="API gốc",
    description="Endpoint kiểm tra để xác nhận API đang hoạt động.",
    response_description="Thông báo xác nhận hệ thống đang hoạt động"
)
async def root():
    return {"message": "Face Recognition Attendance System"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)


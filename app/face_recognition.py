import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from .config import FACE_RECOGNITION_THRESHOLD
from .database import get_last_attendance_status, update_attendance_status
from .attendance import can_record_attendance, log_attendance
from datetime import datetime

# Danh sách provider theo thứ tự ưu tiên
PROVIDER_LIST = [
    'CUDAExecutionProvider',  # Ưu tiên GPU NVIDIA nếu có
    'DmlExecutionProvider',   # DirectML cho Windows
    'CPUExecutionProvider'    # Fallback cuối cùng cho CPU
]

# Hàm khởi tạo FaceAnalysis với fallback
def initialize_face_analyzer():
    """Khởi tạo FaceAnalysis với fallback qua các provider"""
    for provider in PROVIDER_LIST:
        try:
            print(f"Attempting to initialize FaceAnalysis with {provider}...")
            face_analyzer = FaceAnalysis(name='buffalo_l', providers=[provider])
            face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print(f"Successfully initialized with {provider}")
            return face_analyzer
        except Exception as e:
            print(f"Failed to initialize with {provider}: {str(e)}")
    raise RuntimeError("Failed to initialize FaceAnalysis with any provider")

# Khởi tạo mô hình nhận diện khuôn mặt với fallback
face_analyzer = initialize_face_analyzer()

# Từ điển lưu embeddings của tất cả người dùng đã đăng ký
face_database = {}

# Load face database từ SQLite
def load_face_database(user_face_data):
    """Tải face database từ dữ liệu người dùng"""
    face_db = {}
    user_info = {}
    
    for user_id, name, image_path in user_face_data:
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
    
    # Tính trung bình các embedding cho mỗi người dùng
    averaged_db = {}
    for user_key, embeddings in face_db.items():
        if embeddings:
            averaged_embedding = np.mean(embeddings, axis=0)
            averaged_db[user_key] = {
                "embedding": averaged_embedding,
                "info": user_info[user_key]
            }
    
    return averaged_db

def detect_faces(image):
    """Phát hiện khuôn mặt trong ảnh"""
    return face_analyzer.get(image)

def process_frame(frame, face_database):
    """Xử lý frame để nhận diện khuôn mặt"""
    # Copy frame để vẽ lên
    display_frame = frame.copy()
    recognized_users = []
    
    try:
        # Phát hiện khuôn mặt trong frame
        faces = face_analyzer.get(frame)
        
        for face in faces:
            # Lấy embedding của khuôn mặt
            face_embedding = face.normed_embedding

            # Biến lưu thông tin người được nhận diện và độ tương đồng
            match_info, max_similarity = recognize_face(face_embedding, face_database)
            
            # Lấy tọa độ khuôn mặt
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Vẽ kết quả nhận diện lên frame
            draw_recognition_result(display_frame, left, top, right, bottom, 
                                  match_info, max_similarity, recognized_users)
    
    except Exception as e:
        print(f"Lỗi nhận diện: {e}")
    
    return display_frame, recognized_users

# So sánh embedding với face database
def recognize_face(face_embedding, face_database):
    """Nhận diện khuôn mặt dựa trên embedding"""
    match_info = None
    max_similarity = -1
    
    for user_key, data in face_database.items():
        db_embedding = data["embedding"]
        similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > FACE_RECOGNITION_THRESHOLD:
                match_info = data["info"]
            else:
                match_info = None
    
    return match_info, max_similarity

# Vẽ kết quả nhận diện lên frame
def draw_recognition_result(frame, left, top, right, bottom, match_info, similarity, recognized_users):
    """Vẽ kết quả nhận diện lên frame"""
    if match_info:
        # Vẽ khung xanh nếu nhận diện được
        color = (0, 255, 0)
        name = match_info["name"]
        user_id = match_info["user_id"]

        # Hiển thị tên và độ tương đồng
        label = f"{name} ({similarity:.2f})"
        
        # Kiểm tra và ghi nhận điểm danh
        if can_record_attendance(user_id):
            event_type = determine_event_type(user_id)
            log_attendance(name, user_id, event_type)
            recognized_users.append({
                "user_id": user_id,
                "name": name,
                "event": event_type,
                "confidence": float(similarity)
            })
            update_attendance_status(user_id, event_type, datetime.now())
    else:
        # Vẽ khung đỏ nếu không nhận diện được
        color = (0, 0, 255)
        label = f"Unknown ({similarity:.2f})"
    
    # Vẽ bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Hiển thị nhãn
    y_position = max(top - 10, 20) # Đảm bảo text không bị cắt khỏi frame
    cv2.putText(frame, label, (left, y_position),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Xác định loại sự kiện (check-in hoặc check-out)
def determine_event_type(user_id):
    """Xác định loại sự kiện (check-in hoặc check-out)"""
    last_status = get_last_attendance_status(user_id)
    if last_status and last_status[0] == "check-in":
        return "check-out"
    return "check-in"
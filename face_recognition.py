import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime, timedelta

# Cấu hình
DATASET_DIR = "dataset"
DETECTION_THRESHOLD = 0.5  # Ngưỡng nhận diện
MIN_ATTENDANCE_INTERVAL = timedelta(minutes=10)  # Khoảng cách tối thiểu giữa 2 lần điểm danh

# Khởi tạo mô hình nhận diện khuôn mặt
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Từ điển lưu thời gian điểm danh gần nhất
last_attendance_time = {}

def load_face_database(user_data):
    """Tạo face database từ dữ liệu người dùng và ảnh"""
    face_db = {}
    user_info = {}
    
    for user_id, name, image_path in user_data:
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
    
    # Tính trung bình embedding cho mỗi người
    averaged_db = {}
    for user_key, embeddings in face_db.items():
        if embeddings:
            averaged_embedding = np.mean(embeddings, axis=0)
            averaged_db[user_key] = {
                "embedding": averaged_embedding,
                "info": user_info[user_key]
            }
    
    return averaged_db

def can_record_attendance(user_id: str):
    """Kiểm tra xem có thể ghi nhận điểm danh không"""
    current_time = datetime.now()
    if user_id in last_attendance_time:
        last_time = last_attendance_time[user_id]
        if current_time - last_time < MIN_ATTENDANCE_INTERVAL:
            return False
    last_attendance_time[user_id] = current_time
    return True

def process_frame(frame, face_database):
    """Xử lý frame để nhận diện khuôn mặt"""
    display_frame = frame.copy()
    recognized_users = []
    
    try:
        faces = face_analyzer.get(frame)
        for face in faces:
            face_embedding = face.normed_embedding
            match_info = None
            max_similarity = -1
            
            for user_key, data in face_database.items():
                db_embedding = data["embedding"]
                similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    if similarity > DETECTION_THRESHOLD:
                        match_info = data["info"]
                    else:
                        match_info = None
            
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            
            if match_info:
                color = (0, 255, 0)
                name = match_info["name"]
                user_id = match_info["user_id"]
                label = f"{name} ({max_similarity:.2f})"
                
                if can_record_attendance(user_id):
                    recognized_users.append({
                        "user_id": user_id,
                        "name": name,
                        "event": "check-in",
                        "confidence": float(max_similarity)
                    })
            else:
                color = (0, 0, 255)
                label = f"Unknown ({max_similarity:.2f})"
            
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            y_position = max(top - 10, 20)
            cv2.putText(display_frame, label, (left, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    except Exception as e:
        print(f"Error in face recognition: {e}")
    
    return display_frame, recognized_users
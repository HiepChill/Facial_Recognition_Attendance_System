import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Thư mục chứa ảnh database
database_path = "dataset/"
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Khởi tạo mô hình nhận diện khuôn mặt CPU
# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0)
# Khởi tạo mô hình nhận diện khuôn mặt GPU
app = FaceAnalysis(name='buffalo_l', providers=['OpenVINOExecutionProvider'])
app.prepare(ctx_id=0)


# Hàm nạp dữ liệu khuôn mặt từ database
def load_face_database():
    face_db = {}
    for person_name in os.listdir(database_path):
        person_folder = os.path.join(database_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Lỗi đọc ảnh: {image_path}, bỏ qua...")
                    continue
                faces = app.get(img)
                if faces:
                    face_db[person_name] = faces[0].normed_embedding
    return face_db

# Nạp database khuôn mặt
face_database = load_face_database()

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Nhận diện khuôn mặt trong frame
        faces = app.get(frame)
        match_name = "No match found"
        max_similarity = -1  # Giá trị tối đa của cosine similarity
        
        for face in faces:
            face_embedding = face.normed_embedding
            
            # So sánh với database bằng cosine similarity
            for name, db_embedding in face_database.items():
                similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    match_name = name if max_similarity > 0.5 else "Unknown"  # Ngưỡng 0.5
        
        # Hiển thị kết quả
        cv2.putText(frame, match_name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        print("Lỗi nhận diện:", e)
    
    # Hiển thị hình ảnh
    cv2.imshow('InsightFace Recognition (ArcFace)', frame)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
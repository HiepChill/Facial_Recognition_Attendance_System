import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# from anti_spoofing import AntiSpoofingModel #new

app = FastAPI()

# Thư mục chứa ảnh database
database_path = "dataset/"
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Khởi tạo mô hình nhận diện khuôn mặt GPU
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_analyzer.prepare(ctx_id=0)

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
                faces = face_analyzer.get(img)
                if faces:
                    face_db[person_name] = faces[0].normed_embedding
    return face_db

# Nạp database khuôn mặt
face_database = load_face_database()

# Khởi tạo camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Nhận diện khuôn mặt trong frame
            faces = face_analyzer.get(frame)
            
            for face in faces:
                face_embedding = face.normed_embedding
                match_name = "No match found"
                max_similarity = -1  # Giá trị tối đa của cosine similarity
                
                # So sánh với database bằng cosine similarity
                for name, db_embedding in face_database.items():
                    similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        match_name = name if max_similarity > 0.5 else "Unknown"  # Ngưỡng 0.5

                # Lấy tọa độ khuôn mặt
                left, top, right, bottom = map(int, face.bbox)

                # Vẽ khung nhận diện
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Hiển thị tên trên ảnh
                cv2.putText(frame, match_name, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print("Lỗi nhận diện:", e)

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Hiển thị hình ảnh
        # cv2.imshow('InsightFace Recognition (ArcFace)', frame)

    cap.release()
    cv2.destroyAllWindows()

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)


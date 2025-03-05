# # import requests

# # url = "http://127.0.0.1:5000/sum"
# # data = {"num1": 5, "num2": 7}

# # response = requests.post(url, json=data)
# # print(response.json())  # Kết quả: {"sum": 12}

# # import torch

# # if torch.cuda.is_available():
# #     print("GPU đã được kích hoạt:", torch.cuda.get_device_name(0))
# # else:
# #     print("Không tìm thấy GPU!")

# import cv2
# import os
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity

# # Thư mục chứa ảnh database
# database_path = "dataset/"
# if not os.path.exists(database_path):
#     os.makedirs(database_path)

# # Khởi tạo mô hình nhận diện khuôn mặt với GPU
# app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# # Prepare with the correct parameters
# app.prepare(ctx_id=0, det_size=(640, 640))

# # Hàm nạp dữ liệu khuôn mặt từ database
# def load_face_database():
#     face_db = {}
#     for person_name in os.listdir(database_path):
#         person_folder = os.path.join(database_path, person_name)
#         if os.path.isdir(person_folder):
#             for image_name in os.listdir(person_folder):
#                 image_path = os.path.join(person_folder, image_name)
#                 img = cv2.imread(image_path)
#                 if img is None:
#                     print(f"Lỗi đọc ảnh: {image_path}, bỏ qua...")
#                     continue
#                 faces = app.get(img)
#                 if faces:
#                     face_db[person_name] = faces[0].normed_embedding
#     return face_db

# # Nạp database khuôn mặt
# face_database = load_face_database()

# # Khởi tạo camera
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     try:
#         # Nhận diện khuôn mặt trong frame
#         faces = app.get(frame)
        
#         for face in faces:
#             # Lấy tọa độ khuôn mặt
#             left, top, right, bottom = map(int, face.bbox)
            
#             # Kiểm tra thuộc tính face anti-spoofing (nếu có)
#             is_real = True  # Mặc định giả sử là khuôn mặt thật
#             liveness_score = 0.5  # Giá trị mặc định
            
#             # Thử các thuộc tính có thể chứa điểm liveness
#             for attr in ['liveness', 'live', 'anti_spoofing', 'spoof_prob']:
#                 if hasattr(face, attr):
#                     liveness_score = getattr(face, attr)
#                     if isinstance(liveness_score, np.ndarray) and liveness_score.size > 0:
#                         liveness_score = liveness_score[0]
#                     break
            
#             # In ra để debug
#             print(f"Face attributes: {dir(face)}")
#             print(f"Liveness score: {liveness_score}")
            
#             # Kiểm tra xem có phải khuôn mặt thật không (ngưỡng có thể điều chỉnh)
#             is_real = liveness_score > 0.5
                    
#             face_embedding = face.normed_embedding
#             match_name = "No match found"
#             max_similarity = -1
            
#             # So sánh với database bằng cosine similarity
#             for name, db_embedding in face_database.items():
#                 similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
                
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     match_name = name
            
#             if max_similarity < 0.5:
#                 match_name = "Unknown"
            
#             if is_real:
#                 # Vẽ khung xanh cho khuôn mặt thật
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
#                 # Hiển thị tên và điểm liveness
#                 text = f"{match_name} (Real: {liveness_score:.2f})"
#                 cv2.putText(frame, text, (left, max(top - 10, 20)), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             else:
#                 # Vẽ khung đỏ cho khuôn mặt giả (ảnh, tranh, màn hình)
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#                 cv2.putText(frame, f"Fake ({liveness_score:.2f})", 
#                            (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 
#                            0.6, (0, 0, 255), 2)
    
#     except Exception as e:
#         print("Lỗi nhận diện:", e)
        
#     # Hiển thị thông tin về liveness detection
#     info_text = "Khuon mat that (mau xanh), Khuon mat gia (mau do)"
#     cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
#                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
#     # Hiển thị hình ảnh
#     cv2.imshow('InsightFace Recognition (ArcFace)', frame)
    
#     # Thoát nếu nhấn phím 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI()

def generate_frames():
    cap = cv2.VideoCapture(0)  # Mở webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
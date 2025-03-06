import cv2
from deepface import DeepFace
import os

# Thư mục chứa ảnh database
database_path = "dataset/"
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Tìm khuôn mặt trong database
        result = DeepFace.find(frame, db_path=database_path, enforce_detection=False, model_name='ArcFace', detector_backend='opencv', align=True, distance_metric='cosine')
        
        if len(result) > 0 and not result[0].empty:
            match_path = result[0].iloc[0]['identity']  # Đường dẫn ảnh nhận diện gần nhất
            match_name = os.path.basename(os.path.dirname(match_path))  # Lấy tên thư mục chứa ảnh
            text = f"Match: {match_name}"
        else:
            text = "No match found"
        
        # Vẽ lên khung hình
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        print("Lỗi nhận diện:", e)
    
    # Hiển thị hình ảnh
    cv2.imshow('DeepFace Face Recognition', frame)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

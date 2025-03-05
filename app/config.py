import os

# Thiết lập tham số 
FACE_RECOGNITION_THRESHOLD = 0.5
ATTENDANCE_INTERVAL_MINUTES = 5

# Cấu hình đường dẫn
DB_PATH = "attendance.db"
DATASET_DIR = "./dataset"
ATTENDANCE_DIR = "./attendance_logs"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
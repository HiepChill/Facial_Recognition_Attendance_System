import csv
import os
from datetime import datetime, timedelta
from .config import ATTENDANCE_DIR, ATTENDANCE_INTERVAL_MINUTES

# Từ điển lưu thời gian điểm danh gần nhất của mỗi người
last_attendance_time = {}

def log_attendance(name, user_id, event_type):
    """Ghi nhận điểm danh"""
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    
    # Tạo file CSV với header nếu chưa tồn tại
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'User ID', 'Time', 'Event'])
    
    # Ghi thông tin điểm danh
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(csv_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([name, user_id, current_time, event_type])

def can_record_attendance(user_id):
    """Kiểm tra thời gian giữa 2 lần điểm danh"""
    current_time = datetime.now()
    
    if user_id in last_attendance_time:
        last_time = last_attendance_time[user_id]
        time_diff = current_time - last_time
        
        # Nếu chưa đủ thời gian, không ghi nhận
        if time_diff < timedelta(minutes=ATTENDANCE_INTERVAL_MINUTES):
            return False
    
    # Cập nhật thời gian điểm danh gần nhất
    last_attendance_time[user_id] = current_time
    return True

def get_attendance_records(date=None):
    """Lấy dữ liệu điểm danh theo ngày"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
        
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date}.csv")
    
    if not os.path.exists(csv_path):
        return []
    
    records = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua header
        
        for row in reader:
            if len(row) >= 4:
                records.append({
                    "name": row[0],
                    "user_id": row[1],
                    "time": row[2],
                    "event": row[3]
                })
    
    return records
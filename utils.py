import os
import csv
from datetime import datetime

ATTENDANCE_DIR = "attendance_logs"

def log_attendance(name: str, user_id: str, event_type: str):
    """Ghi nhận điểm danh vào file CSV"""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'User ID', 'Time', 'Event'])
    
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, user_id, current_time, event_type])

def get_attendance(date: str):
    """Lấy dữ liệu điểm danh theo ngày"""
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{date}.csv")
    if not os.path.exists(csv_path):
        return []
    
    records = []
    with open(csv_path, 'r') as file:
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
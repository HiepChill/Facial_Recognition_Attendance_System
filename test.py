# import cv2
# import time

# # URL RTSP
# rtsp_url = "rtsp://127.0.0.1:8554/mystream"

# try:
#     print("Đang mở kết nối RTSP...")
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
#     if not cap.isOpened():
#         print(f"Không thể mở kết nối tới {rtsp_url}")
#         exit()
#     else:
#         print(f"Kết nối thành công tới {rtsp_url}")
        
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         cv2.imshow("RTSP Stream", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Delay 1ms
#             break
            
# except Exception as e:
#     print(f"Lỗi: {e}")
# finally:
#     if 'cap' in locals() and cap.isOpened():
#         cap.release()
#     cv2.destroyAllWindows()

import cv2
import subprocess

# Thông tin luồng RTSP
rtsp_url = "rtsp://127.0.0.1:8554/mystream"

# Mở webcam
cap = cv2.VideoCapture(0)

# Kiểm tra nếu webcam không mở được
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Lấy thông tin về độ phân giải và FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Mặc định 30 FPS nếu không đọc được

# Lệnh FFmpeg để gửi luồng RTSP
ffmpeg_cmd = [
    "ffmpeg",
    "-y", "-an",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}",
    "-r", str(fps),
    "-i", "-",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    rtsp_url
]

# Chạy FFmpeg với stdin để truyền frame từ OpenCV
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gửi frame đến FFmpeg
    process.stdin.write(frame.tobytes())

    # Hiển thị frame (tuỳ chọn)
    cv2.imshow("Streaming", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
process.stdin.close()
process.wait()
cv2.destroyAllWindows()
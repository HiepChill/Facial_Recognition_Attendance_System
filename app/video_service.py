import cv2
import time
import queue
import threading
from typing import Dict, Any, Generator
import numpy as np

from app.camera import CameraManager
from app.face_recognition import process_frame
from app.rtsp_service import RTSPStreamService

class VideoStreamService:
    """Service quản lý các luồng video và xử lý streaming"""
    
    def __init__(self):
        """Khởi tạo video stream service"""
        self.camera_manager = CameraManager.get_instance()
        self.rtsp_service = RTSPStreamService()
        self.active_streams = {}  # Lưu trữ các luồng stream đang hoạt động
    
    def generate_frames(self, face_database: Dict[str, Any]) -> Generator[bytes, None, None]:
        """Generator để stream video từ camera mặc định"""
        cap = self.camera_manager.get_camera()
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Không thể đọc frame từ camera")
                    time.sleep(0.5)
                    continue
                
                # Xử lý frame cho nhận diện
                processed_frame, _ = process_frame(frame, face_database)
                
                # Chuyển đổi frame thành JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                # Yield frame cho streaming
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Thêm delay nhỏ để điều chỉnh tốc độ phát
                time.sleep(0.03)  # Khoảng 30fps
        
        except Exception as e:
            print(f"Lỗi stream camera: {e}")
        finally:
            # Không đóng camera ở đây để tránh đóng camera khi vẫn có người dùng khác
            pass
    
    def generate_rtsp_frames(self, rtsp_url: str, face_database: Dict[str, Any]) -> Generator[bytes, None, None]:
        """Generator để stream video từ RTSP với xử lý lỗi cải tiến"""
        # Mở stream RTSP
        stream_info = self.rtsp_service.open_rtsp_stream(rtsp_url)
        
        if not stream_info:
            # Tạo frame thông báo lỗi
            error_frame = self._create_error_frame("Không thể kết nối đến RTSP")
            _, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame lỗi
            for _ in range(10):  # Gửi thông báo lỗi 10 lần
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
            return
        
        stream_id = stream_info["id"]
        no_frame_count = 0
        
        try:
            while True:
                # Lấy frame từ RTSP service
                success, frame = self.rtsp_service.get_frame(stream_id)
                
                if not success or frame is None:
                    no_frame_count += 1
                    
                    # Nếu không nhận được frame sau nhiều lần thử
                    if no_frame_count > 5:
                        # Tạo frame thông báo đang kết nối lại
                        reconnect_frame = self._create_error_frame("Đang kết nối lại RTSP...")
                        _, buffer = cv2.imencode('.jpg', reconnect_frame)
                        frame_bytes = buffer.tobytes()
                        
                        # Yield frame thông báo
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    time.sleep(0.5)
                    continue
                
                # Reset biến đếm khi nhận được frame
                no_frame_count = 0
                
                # Xử lý frame cho nhận diện
                processed_frame, _ = process_frame(frame, face_database)
                
                # Chuyển đổi frame thành JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                # Yield frame cho streaming
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Thêm delay nhỏ để điều chỉnh tốc độ phát
                time.sleep(0.03)  # Khoảng 30fps
        
        except Exception as e:
            print(f"Lỗi stream RTSP: {e}")
            
            # Tạo frame thông báo lỗi
            error_frame = self._create_error_frame(f"Lỗi kết nối: {str(e)}")
            _, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame lỗi
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def _create_error_frame(self, message: str, width: int = 640, height: int = 480):
        """Tạo frame thông báo lỗi"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Đặt font và kích thước
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)  # Màu trắng
        
        # Tính toán vị trí để căn giữa văn bản
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Thêm văn bản vào frame
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame
    
    def cleanup_idle_streams(self, max_idle_time: int = 60):
        """Dọn dẹp các stream không hoạt động"""
        self.rtsp_service.cleanup_idle_streams(max_idle_time)

# Tạo instance duy nhất
_video_service_instance = None

def get_video_service():
    """Trả về instance duy nhất của VideoStreamService"""
    global _video_service_instance
    if _video_service_instance is None:
        _video_service_instance = VideoStreamService()
    return _video_service_instance
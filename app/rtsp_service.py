import cv2
import time
import queue
import threading
import numpy as np
from typing import Dict, Any, Generator, Optional, Tuple

class RTSPStreamService:
    """Service chuyên xử lý luồng RTSP"""
    
    def __init__(self):
        """Khởi tạo RTSP stream service"""
        self.streams = {}  # Lưu trữ các luồng stream đang hoạt động
    
    def open_rtsp_stream(self, rtsp_url: str) -> Optional[Dict]:
        """Mở và chuẩn bị stream RTSP"""
        stream_id = f"rtsp_{hash(rtsp_url)}"
        
        # Kiểm tra xem stream đã tồn tại chưa
        if stream_id in self.streams and not self.streams[stream_id].get("error", False):
            # Cập nhật thời gian truy cập
            self.streams[stream_id]["last_access"] = time.time()
            return self.streams[stream_id]
        
        # Thử mở kết nối mới với RTSP
        try:
            # Sử dụng các tham số đặc biệt để cải thiện hiệu suất và độ ổn định
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
            # Thiết lập các thuộc tính để tăng độ ổn định
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Đặt kích thước buffer
            
            # Kiểm tra kết nối
            if not cap.isOpened():
                print(f"Không thể mở RTSP: {rtsp_url}")
                return None
            
            # Khởi tạo buffer và các thông tin cần thiết
            frame_buffer = queue.Queue(maxsize=30)  # Lưu tối đa 30 frame
            stop_event = threading.Event()
            
            # Tạo thông tin stream
            stream_info = {
                "id": stream_id,
                "rtsp_url": rtsp_url,
                "cap": cap,
                "buffer": frame_buffer,
                "stop_event": stop_event,
                "last_access": time.time(),
                "last_frame": None,
                "error": False,
                "error_count": 0,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0
            }
            
            # Lưu thông tin stream
            self.streams[stream_id] = stream_info
            
            # Bắt đầu thread đọc frame
            capture_thread = threading.Thread(target=self._capture_frames, args=(stream_id,))
            capture_thread.daemon = True
            capture_thread.start()
            
            # Lưu thread
            self.streams[stream_id]["thread"] = capture_thread
            
            return stream_info
            
        except Exception as e:
            print(f"Lỗi khi mở RTSP {rtsp_url}: {e}")
            return None
    
    def _capture_frames(self, stream_id: str):
        """Luồng chạy ngầm để ghi nhận frame từ RTSP"""
        if stream_id not in self.streams:
            return
            
        stream_info = self.streams[stream_id]
        cap = stream_info["cap"]
        frame_buffer = stream_info["buffer"]
        stop_event = stream_info["stop_event"]
        rtsp_url = stream_info["rtsp_url"]
        
        consecutive_errors = 0
        retry_count = 0
        
        try:
            while not stop_event.is_set():
                try:
                    success, frame = cap.read()
                    
                    if not success:
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            # Thử kết nối lại sau một số lần lỗi liên tiếp
                            print(f"Mất kết nối với RTSP {rtsp_url}. Đang thử kết nối lại...")
                            cap.release()
                            time.sleep(2)  # Đợi 2 giây trước khi thử lại
                            
                            # Thử kết nối lại
                            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                            
                            if not cap.isOpened():
                                retry_count += 1
                                if retry_count > 3:  # Nếu thử lại 3 lần không thành công
                                    stream_info["error"] = True
                                    print(f"Không thể kết nối lại với RTSP {rtsp_url} sau 3 lần thử")
                                    break
                            else:
                                # Cập nhật cap trong stream_info
                                self.streams[stream_id]["cap"] = cap
                                consecutive_errors = 0
                                retry_count = 0
                        
                        time.sleep(0.1)  # Đợi một chút trước khi thử lại
                        continue
                    
                    # Reset biến đếm lỗi khi đọc thành công
                    consecutive_errors = 0
                    
                    # Kiểm tra frame có hợp lệ không
                    if frame is None or frame.size == 0:
                        continue
                    
                    # Lưu frame cuối cùng (để trường hợp buffer rỗng)
                    stream_info["last_frame"] = frame.copy()
                    
                    # Đưa vào buffer, loại bỏ frame cũ nếu đầy
                    if frame_buffer.full():
                        try:
                            frame_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    frame_buffer.put(frame)
                    
                    # Thêm một chút delay để giảm tải CPU
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Lỗi khi đọc frame từ RTSP {rtsp_url}: {e}")
                    consecutive_errors += 1
                    time.sleep(0.5)
            
        except Exception as e:
            print(f"Lỗi thread ghi nhận RTSP {rtsp_url}: {e}")
        finally:
            # Đánh dấu stream bị lỗi
            if stream_id in self.streams:
                self.streams[stream_id]["error"] = True
                
                # Đóng camera
                try:
                    cap.release()
                except:
                    pass
    
    def get_frame(self, stream_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Lấy frame từ stream"""
        if stream_id not in self.streams:
            return False, None
            
        stream_info = self.streams[stream_id]
        
        # Cập nhật thời gian truy cập
        stream_info["last_access"] = time.time()
        
        # Kiểm tra nếu stream bị lỗi
        if stream_info.get("error", False):
            return False, None
        
        # Lấy frame từ buffer
        try:
            frame_buffer = stream_info["buffer"]
            
            if not frame_buffer.empty():
                return True, frame_buffer.get()
            elif stream_info["last_frame"] is not None:
                # Trả về frame cuối cùng nếu buffer rỗng
                return True, stream_info["last_frame"]
            else:
                return False, None
                
        except Exception as e:
            print(f"Lỗi khi lấy frame từ buffer: {e}")
            return False, None
    
    def close_stream(self, stream_id: str):
        """Đóng stream"""
        if stream_id in self.streams:
            try:
                # Thiết lập cờ dừng
                self.streams[stream_id]["stop_event"].set()
                
                # Đợi thread dừng
                if "thread" in self.streams[stream_id]:
                    self.streams[stream_id]["thread"].join(timeout=1.0)
                
                # Đóng camera
                if "cap" in self.streams[stream_id]:
                    self.streams[stream_id]["cap"].release()
                
                # Xóa stream
                del self.streams[stream_id]
                print(f"Đã đóng stream {stream_id}")
                
            except Exception as e:
                print(f"Lỗi khi đóng stream {stream_id}: {e}")
    
    def cleanup_idle_streams(self, max_idle_time: int = 60):
        """Dọn dẹp các stream không hoạt động"""
        current_time = time.time()
        streams_to_close = []
        
        for stream_id, stream_info in self.streams.items():
            idle_time = current_time - stream_info["last_access"]
            if idle_time > max_idle_time or stream_info.get("error", False):
                streams_to_close.append(stream_id)
        
        for stream_id in streams_to_close:
            self.close_stream(stream_id)
# app/camera.py
import cv2
import threading

class CameraManager:
    _instance = None
    _camera = None
    _lock = threading.Lock()
    _rtsp_url = None  # Biến lưu URL RTSP
    
    @classmethod
    def get_instance(cls):
        """Lấy instance của CameraManager (Singleton pattern)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Khởi tạo CameraManager"""
        if CameraManager._instance is not None:
            raise Exception("Singleton class - sử dụng get_instance()")
        self._camera = None
        self._rtsp_url = None
    
    def get_camera(self, rtsp_url=None):
        """Lấy camera, khởi tạo nếu chưa có, hỗ trợ RTSP"""
        if rtsp_url:
            # Nếu có RTSP URL, sử dụng nó
            if self._rtsp_url != rtsp_url or self._camera is None:
                self.release_camera()
                self._rtsp_url = rtsp_url
                self._camera = cv2.VideoCapture(rtsp_url)
        else:
            # Nếu không có RTSP URL, dùng camera native
            if self._camera is None or self._rtsp_url is not None:
                self.release_camera()
                self._camera = cv2.VideoCapture(0)
                self._rtsp_url = None
        
        if not self._camera.isOpened():
            raise Exception("Không thể mở camera hoặc kết nối RTSP")
        return self._camera
    
    def release_camera(self):
        """Giải phóng tài nguyên camera"""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            self._rtsp_url = None
            print("Camera đã được giải phóng")
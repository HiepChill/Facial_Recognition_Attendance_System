import cv2
import threading

class CameraManager:
    _instance = None
    _camera = None
    _lock = threading.Lock()
    
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
    
    def get_camera(self):
        """Lấy camera, khởi tạo nếu chưa có"""
        if self._camera is None:
            self._camera = cv2.VideoCapture(0)
        return self._camera
    
    def release_camera(self):
        """Giải phóng tài nguyên camera"""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            print("Camera đã được giải phóng")
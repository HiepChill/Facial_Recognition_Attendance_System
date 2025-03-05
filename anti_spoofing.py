import torch
import torch.nn as nn
import cv2
import numpy as np
from miniFASNet import MiniFASNetV2  # Import kiến trúc mạng

class AntiSpoofingModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Khởi tạo mô hình MiniFASNet
        self.model = MiniFASNetV2(num_classes=1).to(self.device)

        # Load trọng số mô hình
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)  # Cập nhật trọng số

        self.model.eval()  # Chuyển sang chế độ đánh giá
    
    def preprocess(self, img):
        img = cv2.resize(img, (80, 80))
        img = img.astype(np.float32) / 255.0  # Chuẩn hóa pixel
        img = np.transpose(img, (2, 0, 1))  # Chuyển thành (C, H, W)
        img = np.expand_dims(img, axis=0)  # Thêm batch dimension
        img = torch.from_numpy(img).float().to(self.device)
        return img
    
    def predict(self, img):
        img_tensor = self.preprocess(img)
        with torch.no_grad():
            output = self.model(img_tensor)
        score = torch.sigmoid(output).cpu().numpy().flatten()[0]
        return score  # Giá trị càng cao, khả năng là giả càng lớn

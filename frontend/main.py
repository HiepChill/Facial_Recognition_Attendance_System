import sys
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap

API_URL = "http://127.0.0.1:5000"

class FaceAttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance")
        self.setGeometry(100, 100, 600, 400)

        self.label = QLabel("Chọn ảnh để nhận diện:", self)
        self.label.setStyleSheet("font-size: 16px;")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)

        self.upload_button = QPushButton("Chọn ảnh", self)
        self.upload_button.clicked.connect(self.upload_image)

        self.result_label = QLabel("", self)
        self.result_label.setStyleSheet("font-size: 14px; color: green;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return

        self.image_label.setPixmap(QPixmap(file_path).scaled(300, 300))
        
        with open(file_path, "rb") as file:
            response = requests.post(f"{API_URL}/attendance/recognize", files={"image": file})

        if response.status_code == 200:
            self.result_label.setText(response.json()["message"])
        else:
            self.result_label.setText("Không nhận diện được khuôn mặt.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceAttendanceApp()
    window.show()
    sys.exit(app.exec_())

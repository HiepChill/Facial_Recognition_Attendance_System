# Face Recognition Attendance System

## Tổng quan

**Face Recognition Attendance System** là một ứng dụng điểm danh tự động sử dụng công nghệ nhận diện khuôn mặt. Dự án được xây dựng bằng Python, sử dụng thư viện **InsightFace** để nhận diện khuôn mặt và **FastAPI** để tạo API phục vụ giao tiếp với người dùng. Hệ thống hỗ trợ cả **check-in** và **check-out**, lưu trữ dữ liệu điểm danh vào file CSV và cơ sở dữ liệu SQLite, đồng thời cho phép stream video trực tiếp để nhận diện khuôn mặt theo thời gian thực.

Hệ thống phù hợp cho các ứng dụng như quản lý điểm danh nhân viên, sinh viên, hoặc các sự kiện cần kiểm soát ra vào.

---

## Tính năng chính

-   **Đăng ký người dùng**: Đăng ký người dùng mới với nhiều ảnh khuôn mặt và lưu trữ vào cơ sở dữ liệu.
-   **Nhận diện khuôn mặt**: Sử dụng InsightFace để nhận diện khuôn mặt trong video stream hoặc ảnh tĩnh.
-   **Điểm danh tự động**: Hỗ trợ cả sự kiện **check-in** và **check-out**, với khoảng thời gian tối thiểu giữa hai lần điểm danh (mặc định là 10 phút).
-   **Fallback Providers**: Hỗ trợ nhiều provider cho InsightFace (`CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider`) để đảm bảo tương thích với các môi trường phần cứng khác nhau.
-   **Lưu trữ dữ liệu**:
    -   Dữ liệu người dùng và ảnh khuôn mặt được lưu trong SQLite.
    -   Lịch sử điểm danh được lưu dưới dạng file CSV trong thư mục `attendance_logs`.
-   **API RESTful**:
    -   Đăng ký người dùng (`/register_face`).
    -   Xem danh sách người dùng (`/users`).
    -   Xem ảnh khuôn mặt của người dùng (`/user/{user_id}/faces`).
    -   Xem lịch sử điểm danh theo ngày (`/attendance/{date}`) hoặc ngày hiện tại (`/today_attendance`).
    -   Stream video nhận diện khuôn mặt theo thời gian thực (`/video_feed`).

---

## Cấu trúc thư mục

```
📂face_recognition_system/
├── app/
│ ├── init.py # Định nghĩa module và import
│ ├── attendance.py # Logic xử lý điểm danh (lưu và lấy dữ liệu điểm danh)
│ ├── camera.py # Quản lý camera (singleton pattern)
│ ├── config.py # Cấu hình đường dẫn và tham số
│ ├── database.py # Xử lý cơ sở dữ liệu SQLite
│ ├── face_recognition.py # Logic nhận diện khuôn mặt và xử lý embedding
│ ├── models.py # Định nghĩa các model Pydantic
│ └── main.py # File chính chứa FastAPI app và API endpoints
├── attendance_logs/ # Thư mục lưu lịch sử điểm danh (file CSV)
├── dataset/ # Thư mục lưu ảnh khuôn mặt của người dùng
├── .gitattributes # Cấu hình Git
├── attendance.db # Cơ sở dữ liệu SQLite
├── run.py # File chạy ứng dụng (không cần thiết nếu dùng main.py)
└── README.md # Tài liệu hướng dẫn (file này)
```

---

## Yêu cầu cài đặt

### Phần mềm

-   Python 3.8 hoặc cao hơn
-   Hệ điều hành: Windows, Linux, hoặc macOS
-   Camera (tích hợp hoặc USB) để stream video nhận diện khuôn mặt

### Thư viện Python

Danh sách các thư viện cần thiết được liệt kê trong file `requirements.txt` (tạo file này nếu chưa có):

```
fastapi==0.95.1
uvicorn==0.21.1
opencv-python==4.7.0
numpy==1.24.3
insightface[onnxruntime-gpu]==0.7.3 # Hỗ trợ GPU nếu có
scikit-learn==1.2.2
```

---

### Cài đặt

1. **Clone hoặc tải dự án**:

    ```bash
    git clone https://github.com/HiepChill/Facial_Recognition_Attendance_System.git
    cd face_recognition_system

    ```

2. Tạo môi trường ảo (khuyến nghị):

```
   python -m venv venv
   source venv/bin/activate # Linux/macOS
   venv\Scripts\activate # Windows
```

3. Cài đặt các thư viện:

```
   pip install -r requirements.txt
```

4. (Tùy chọn) Cài đặt CUDA nếu muốn dùng GPU:
   Cài đặt CUDA Toolkit và cuDNN nếu bạn muốn sử dụng CUDAExecutionProvider.
   Đảm bảo phiên bản insightface[onnxruntime-gpu] tương thích với CUDA.

Cách sử dụng

1. Chạy ứng dụng

-   Chạy file main.py để khởi động server FastAPI:

    ```
    python main.py
    ```

    -   Server sẽ chạy tại http://127.0.0.1:8080.
    -   Truy cập http://127.0.0.1:8080/docs để xem tài liệu API tự động từ FastAPI (Swagger UI).

2. Đăng ký người dùng

-   Sử dụng endpoint /register_face để đăng ký người dùng mới:

    -   Phương thức: POST
    -   Body:

    ```
        user_id (Form): ID duy nhất của người dùng (text).
        name (Form): Tên người dùng (text).
        face_images (File): Danh sách file ảnh khuôn mặt (upload nhiều file).
    ```

    -   Ví dụ (dùng curl):

    ```
        curl -X POST "http://127.0.0.1:8080/register_face" \
         -F "user_id=johndoe123" \
         -F "name=John Doe" \
         -F "face_images=@/path/to/face1.jpg" \
         -F "face_images=@/path/to/face2.jpg"
    ```

    -   Phản hồi:

    ```
        {
            "message": "Đăng ký thành công",
            "user_id": "johndoe123",
            "name": "John Doe",
            "image_count": 2
        }
    ```

3. Xem danh sách người dùng:

    - Endpoint: /users (GET)
    - Phản hồi:

    ```
      {
        "users": [
            {"id": "johndoe123", "name": "John Doe"},
            {"id": "janesmith456", "name": "Jane Smith"}
        ]
      }
    ```

4. Xem lịch sử điểm danh

-   Endpoint: /attendance/{date} (GET, date có định dạng YYYY-MM-DD)
-   Ví dụ: /attendance/2025-03-05
-   Phản hồi:
    ```
    {
        "date": "2025-03-05",
        "records": [
            {"name": "John Doe", "user_id": "johndoe123", "time": "08:00:00", "event": "check-in"},
            {"name": "John Doe", "user_id": "johndoe123", "time": "12:00:00", "event": "check-out"}
        ]
    }
    ```
-   Endpoint: /today_attendance (GET) để xem điểm danh hôm nay.

5. Stream video nhận diện khuôn mặt

-   Endpoint: /video_feed (GET)
-   Mở trình duyệt và truy cập http://127.0.0.1:8080/video_feed để xem video stream nhận diện khuôn mặt theo thời gian thực.
-   Hệ thống sẽ tự động ghi nhận điểm danh (check-in/check-out) khi nhận diện được khuôn mặt.

---

Chi tiết kỹ thuật

1. Công nghệ sử dụng
   FastAPI: Framework xây dựng API RESTful.
   InsightFace: Thư viện nhận diện khuôn mặt với mô hình buffalo_l.
   OpenCV: Xử lý video và hình ảnh.
   SQLite: Cơ sở dữ liệu để lưu thông tin người dùng và ảnh khuôn mặt.
   Pydantic: Định nghĩa và xác thực dữ liệu (model cho API).
2. Logic điểm danh
   Hệ thống sử dụng cosine similarity để so sánh embedding khuôn mặt với cơ sở dữ liệu.
   Ngưỡng nhận diện (FACE_RECOGNITION_THRESHOLD) được đặt là 0.5.
   Khoảng thời gian tối thiểu giữa hai lần điểm danh (ATTENDANCE_INTERVAL_MINUTES) là 10 phút.
   Sự kiện check-in và check-out được xác định dựa trên trạng thái trước đó:
   Nếu lần trước là "check-in", lần này sẽ là "check-out", và ngược lại.
3. Fallback Providers
   Hệ thống hỗ trợ nhiều provider cho InsightFace:
   CUDAExecutionProvider: Ưu tiên nếu có GPU NVIDIA và CUDA.
   DmlExecutionProvider: DirectML cho Windows.
   CPUExecutionProvider: Fallback cuối cùng nếu không có GPU.
   Khi khởi động, ứng dụng sẽ thử từng provider theo thứ tự ưu tiên và in log để báo trạng thái.
4. Cấu trúc cơ sở dữ liệu
   users: Lưu thông tin người dùng (id, name, created_at).
   face_images: Lưu đường dẫn ảnh khuôn mặt (id, user_id, image_path, created_at).
   attendance_status: Lưu trạng thái điểm danh gần nhất (user_id, last_event, last_time).

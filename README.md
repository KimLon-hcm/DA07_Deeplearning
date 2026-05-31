# Hướng dẫn Cài đặt và Chạy Dự án (Corn Leaf Disease Classification)

Dự án này bao gồm hai phần chính: Backend (FastAPI + TensorFlow) và Frontend (React + Vite). Dưới đây là hướng dẫn chi tiết để thiết lập và chạy dự án khi thành viên khác clone về máy.

## Yêu cầu hệ thống
- Python 3.8 trở lên
- Node.js 16 trở lên
- Git

## 1. Clone dự án
Mở terminal và chạy lệnh sau để clone dự án về máy:
```bash
git clone <URL_CỦA_REPO>
cd DA07_Deeplearning
```

## 2. Thiết lập và Chạy Backend

Backend được viết bằng Python (FastAPI) và sử dụng thư viện TensorFlow để load model dự đoán.

*Lưu ý:* Mã nguồn backend sẽ tự động load file model từ đường dẫn `Model/ViT_Scratch_best.keras` tính từ thư mục gốc của dự án. Hãy chắc chắn file model này tồn tại trước khi chạy.

###  Mở một terminal
### Bước 1: Di chuyển vào thư mục backend

```bash
cd Source_code/backend
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị)
**Trên Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Chạy server FastAPI
```bash
uvicorn main:app --reload


## 3. Thiết lập và Chạy Frontend

Frontend được phát triển bằng ReactJS với Vite và Tailwind CSS.

### Bước 1: Mở một terminal mới (giữ terminal backend chạy) và vào thư mục frontend
```bash
cd Source_code/frontend
```

### Bước 2: Cài đặt các gói phụ thuộc
```bash
npm install
```

### Bước 3: Chạy giao diện (Development Server)
```bash
npm run dev

'''
xog trên màn hình terminal hiển thị 1 cái link localhost thì ctrl + click vào link đó sẽ mở giao diện web
'''

## Cấu trúc thư mục quan trọng
- `Source_code/backend/`: Chứa mã nguồn của API Server.
- `Source_code/frontend/`: Chứa mã nguồn giao diện người dùng.
- `Model/`: Thư mục lưu trữ model đã được train (ví dụ: `ViT_Scratch_best.keras`).
- `Dataset/`: Chứa dữ liệu ảnh (nếu có trong quá trình train).

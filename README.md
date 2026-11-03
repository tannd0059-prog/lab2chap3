# BÀI THỰC HÀNH CHƯƠNG 3: HUẤN LUYỆN MẠNG NƠ-RON HỌC SÂU (PYTORCH)
**Học phần:** Deep Learning / Machine Learning — Trường ĐH Giao Thông Vận Tải TP.HCM (UTH)  
**Giảng viên:** PhD. Nguyễn Thị Khánh Tiên

---

## 📌 Nội dung & Mục tiêu (Dựa theo Slide 23 - Practice 2)
1. **Thực nghiệm với nhiều pre-trained models**: ResNet-18, VGG-16, DenseNet-121.
2. **Cấu hình & Tinh chỉnh mô hình**:
   - **Transfer Learning (Feature Extraction)**: Đóng băng (`requires_grad = False`) các tầng pre-trained, chỉ huấn luyện classifier mới.
   - **Fine-Tuning**: Mở khóa (`requires_grad = True`) một số tầng cuối cùng để tinh chỉnh trọng số cùng classifier.
3. **Thử nghiệm siêu tham số (Hyperparameters)**: Learning rate, Batch size, Optimizer (Adam vs SGD), Regularization (Weight Decay).
4. **Theo dõi quá trình huấn luyện bằng TensorBoard**: Giám sát đồ thị hàm mất mát (Loss), độ chính xác (Accuracy), ảnh mẫu đầu vào và histograms trọng số.
5. **Lưu & Tải mô hình**: `torch.save` và `torch.load`.

---

## 📂 Cấu trúc thư mục dự án

```text
practice_chapter3/
├── models.py                # Khởi tạo ResNet18, VGG16, DenseNet121 & cấu hình freeze/unfreeze
├── dataset.py               # Chuẩn bị dữ liệu CIFAR-10 & DataLoaders (có chế độ fallback offline)
├── train.py                 # Script huấn luyện với đầy đủ chức năng CLI & TensorBoard
├── practice_chapter3.ipynb  # Jupyter Notebook chi tiết, có thể chạy từng bước
├── runs/                    # Thư mục lưu trữ log của TensorBoard
└── README.md                # Hướng dẫn chi tiết
```

---

## 🚀 Hướng dẫn thực thi

### 1. Huấn luyện qua dòng lệnh (CLI với `train.py`)

#### a. Thử nghiệm Transfer Learning với ResNet-18:
```bash
python train.py --model resnet18 --mode transfer_learning --optimizer adam --lr 0.001 --batch_size 64 --epochs 5 --save_model
```

#### b. Thử nghiệm Fine-Tuning với ResNet-18:
```bash
python train.py --model resnet18 --mode fine_tune --optimizer sgd --lr 0.0001 --batch_size 32 --epochs 5
```

#### c. So sánh với các mô hình khác (VGG-16 hoặc DenseNet-121):
```bash
# VGG-16
python train.py --model vgg16 --mode transfer_learning --lr 0.0005 --epochs 3

# DenseNet-121
python train.py --model densenet121 --mode fine_tune --lr 0.0001 --epochs 3
```

#### d. Chạy thử nghiệm nhanh offline (dùng dummy dataset):
```bash
python train.py --model resnet18 --dataset dummy --epochs 2 --batch_size 16
```

---

### 2. Xem trực quan hóa trên TensorBoard (Slide 13 - 15)

Khởi động TensorBoard bằng lệnh:
```bash
tensorboard --logdir=runs
```
Sau đó mở trình duyệt web và truy cập địa chỉ:
👉 **[http://localhost:6006](http://localhost:6006)**

Trên giao diện TensorBoard, bạn có thể:
- **Scalars**: So sánh đường cong Loss và Accuracy giữa các lần thử nghiệm (ResNet vs VGG, Adam vs SGD, v.v.).
- **Images**: Xem ảnh mẫu được đưa vào mô hình.
- **Graphs**: Xem đồ thị kiến trúc mạng tính toán.
- **Histograms**: Quan sát sự phân bố trọng số và gradient của các lớp theo từng epoch.

---

### 3. Thực hành tương tác trên Jupyter Notebook
Nếu bạn sử dụng VS Code, Jupyter Lab hoặc Google Colab, bạn có thể mở trực tiếp tệp:
`practice_chapter3.ipynb` để chạy và quan sát kết quả từng ô lệnh (code cell).

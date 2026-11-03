import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class DummyImageDataset(Dataset):
    """Dataset giả lập dùng để chạy thử nghiệm và kiểm tra code khi không có kết nối internet."""
    def __init__(self, num_samples=200, img_size=(3, 64, 64), num_classes=10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        # Tạo dữ liệu ngẫu nhiên
        self.data = torch.randn(num_samples, *img_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloaders(dataset_name='cifar10', data_dir='./data', batch_size=32, img_size=64, num_workers=0):
    """
    Chuẩn bị DataLoader cho tập huấn luyện và kiểm thử.
    
    Args:
        dataset_name (str): 'cifar10' hoặc 'dummy'
        data_dir (str): Đường dẫn lưu trữ dữ liệu
        batch_size (int): Kích thước batch (ví dụ 32, 64)
        img_size (int): Kích thước ảnh đầu vào (resizing)
        num_workers (int): Số luồng tải dữ liệu (0 cho Windows để tránh lỗi multiprocessing)
    """
    # Chuẩn hóa ảnh theo thông số chuẩn ImageNet của các mô hình pre-trained
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck')

    if dataset_name.lower() == 'cifar10':
        try:
            print("Đang tải/đọc dataset CIFAR-10...")
            train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
            print(f"CIFAR-10 sẵn sàng: {len(train_dataset)} ảnh train, {len(test_dataset)} ảnh test.")
        except Exception as e:
            print(f"[Cảnh báo] Không thể tải CIFAR-10 tự động ({e}). Chuyển sang Dummy Dataset giả lập để kiểm tra mã nguồn.")
            train_dataset = DummyImageDataset(num_samples=200, img_size=(3, img_size, img_size), num_classes=10)
            test_dataset = DummyImageDataset(num_samples=50, img_size=(3, img_size, img_size), num_classes=10)
    else:
        train_dataset = DummyImageDataset(num_samples=200, img_size=(3, img_size, img_size), num_classes=10)
        test_dataset = DummyImageDataset(num_samples=50, img_size=(3, img_size, img_size), num_classes=10)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, cifar_classes

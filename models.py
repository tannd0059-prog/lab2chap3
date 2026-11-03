import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_model(model_name: str = 'resnet18', num_classes: int = 10, mode: str = 'transfer_learning', pretrained: bool = True):
    """
    Tải mô hình pre-trained (hoặc kiến trúc mô hình) và cấu hình lại classifier cho bài toán phân loại num_classes lớp.
    
    Args:
        model_name (str): 'resnet18', 'vgg16', hoặc 'densenet121'
        num_classes (int): Số lượng class đầu ra (mặc định 10 cho CIFAR-10)
        mode (str): 
            - 'transfer_learning' (Feature Extraction): đóng băng toàn bộ backbone, chỉ train classifier mới.
            - 'fine_tune': unfreeze các layer cuối + classifier để cập nhật trọng số.
        pretrained (bool): True để tải trọng số pre-trained (cần internet lần đầu), False nếu dùng kiến trúc thuần không tải weights.
    """
    model_name = model_name.lower()
    
    # 1. Khởi tạo mô hình
    if model_name == 'resnet18':
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except Exception:
            # Fallback nếu mạng offline hoặc phiên bản torchvision cũ
            model = models.resnet18(weights=None)
            
    elif model_name == 'vgg16':
        try:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
        except Exception:
            model = models.vgg16(weights=None)
            
    elif model_name == 'densenet121':
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
        except Exception:
            model = models.densenet121(weights=None)
            
    else:
        raise ValueError(f"Mô hình '{model_name}' chưa được hỗ trợ. Hãy chọn 'resnet18', 'vgg16', hoặc 'densenet121'.")

    # 2. Xử lý đóng băng (Freeze parameters) theo mode
    if mode == 'transfer_learning':
        # Đóng băng toàn bộ trọng số mạng đã pre-train
        for param in model.parameters():
            param.requires_grad = False
            
    elif mode == 'fine_tune':
        # Đóng băng các tầng đầu, chỉ unfreeze tầng cuối
        for param in model.parameters():
            param.requires_grad = False
            
        if model_name == 'resnet18':
            # Unfreeze layer 4 (khối residual cuối cùng)
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif model_name == 'vgg16':
            # Unfreeze khối conv cuối cùng (từ index 24 trở đi trong features)
            for param in model.features[24:].parameters():
                param.requires_grad = True
        elif model_name == 'densenet121':
            # Unfreeze denseblock4
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True

    # 3. Thay thế classification layer cuối cùng phù hợp với số lớp cần phân loại
    if model_name == 'resnet18':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'vgg16':
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'densenet121':
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model

def print_trainable_parameters(model):
    """In ra danh sách các layer và số tham số cần huấn luyện (trainable vs frozen)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("-" * 55)
    print(f"Tổng số tham số: {total_params:,}")
    print(f"Số tham số huấn luyện (Trainable): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Số tham số đóng băng (Frozen): {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print("-" * 55)

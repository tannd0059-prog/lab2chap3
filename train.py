import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from models import get_pretrained_model, print_trainable_parameters
from dataset import get_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 1. Chuyển dữ liệu lên GPU/CPU
        inputs, targets = inputs.to(device), targets.to(device)

        # 2. Xóa gradient cũ
        optimizer.zero_grad()

        # 3. Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 4. Backward pass (tính gradient)
        loss.backward()

        # 5. Cập nhật trọng số
        optimizer.step()

        # Thống kê
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    elapsed = time.time() - start_time

    return epoch_loss, epoch_acc, elapsed


def evaluate(model, dataloader, criterion, device):
    """Đánh giá mô hình trên tập validation/test."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Tắt tính gradient để tiết kiệm bộ nhớ và tăng tốc
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Chapter 3 Practice: Training Deep Neural Networks with PyTorch")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vgg16', 'densenet121'],
                        help="Tên pre-trained model cần thử nghiệm (ResNet18, VGG16, DenseNet121)")
    parser.add_argument('--mode', type=str, default='transfer_learning', choices=['transfer_learning', 'fine_tune'],
                        help="Chế độ huấn luyện: transfer_learning (chỉ train classifier) hoặc fine_tune (unfreeze các tầng cuối)")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'dummy'],
                        help="Bộ dữ liệu: CIFAR-10 hoặc Dummy Dataset cho offline test")
    parser.add_argument('--epochs', type=int, default=3, help="Số lượng epoch huấn luyện")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=0.001, help="Tốc độ học (learning rate)")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help="Thuật toán tối ưu hóa")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 Regularization (weight decay)")
    parser.add_argument('--img_size', type=int, default=64, help="Kích thước ảnh đầu vào (resizing)")
    parser.add_argument('--exp_name', type=str, default=None, help="Tên thí nghiệm lưu trong TensorBoard")
    parser.add_argument('--save_model', action='store_true', help="Lưu checkpoint mô hình sau khi train")

    args = parser.parse_args()

    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== SỬ DỤNG THIẾT BỊ: {device} ===")

    # Tên thí nghiệm TensorBoard
    if not args.exp_name:
        args.exp_name = f"{args.model}_{args.mode}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}"
    log_dir = os.path.join("runs", args.exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log lưu tại: {log_dir}")

    # 1. Chuẩn bị dữ liệu
    train_loader, test_loader, classes = get_dataloaders(
        dataset_name=args.dataset,
        data_dir="./data",
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=0
    )

    # 2. Khởi tạo Model
    print(f"Khởi tạo mô hình '{args.model}' ở chế độ '{args.mode}'...")
    model = get_pretrained_model(
        model_name=args.model,
        num_classes=len(classes),
        mode=args.mode,
        pretrained=True
    )
    model = model.to(device)
    print_trainable_parameters(model)

    # Ghi nhận sample batch ảnh lên TensorBoard
    sample_images, sample_labels = next(iter(train_loader))
    img_grid = vutils.make_grid(sample_images[:16], normalize=True)
    writer.add_image('Sample_Training_Images', img_grid)

    # Ghi nhận Graph kiến trúc mô hình lên TensorBoard
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    try:
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"[Cảnh báo] Bỏ qua add_graph do: {e}")

    # 3. Loss function và Optimizer
    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print(f"\nBắt đầu huấn luyện {args.epochs} epochs...")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc (%)':<14} | {'Val Loss':<12} | {'Val Acc (%)':<14} | {'Thời gian (s)':<12}")
    print("-" * 80)

    best_val_acc = 0.0

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, elapsed = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer
        )

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # Ghi các thông số lên TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Ghi histogram trọng số các layer lên TensorBoard
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(name, param, epoch)

        print(f"{epoch:<8} | {train_loss:<12.4f} | {train_acc:<14.2f} | {val_loss:<12.4f} | {val_acc:<14.2f} | {elapsed:<12.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_model:
                os.makedirs("checkpoints", exist_ok=True)
                save_path = os.path.join("checkpoints", f"best_{args.model}_{args.mode}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"   -> Đã lưu mô hình tốt nhất ({best_val_acc:.2f}%) tại {save_path}")

    # Đóng TensorBoard writer
    writer.close()
    print("-" * 80)
    print(f"Huấn luyện hoàn tất! Kết quả Validation tốt nhất: {best_val_acc:.2f}%")
    print(f"Để xem trực quan hóa, mở terminal và gõ: tensorboard --logdir=runs\n")

if __name__ == '__main__':
    main()

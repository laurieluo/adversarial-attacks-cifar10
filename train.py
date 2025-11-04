# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse 
from tqdm import tqdm

from src.models import ResNet18, VGG16_BN, DenseNet121
from src.data_loader import get_cifar10_loaders
from src.utils import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10")
    parser.add_argument(
        '--model', 
        type=str, 
        default='resnet18', 
        choices=['resnet18', 'vgg16', 'densenet121'],
        help="Model architecture to train (default: resnet18)"
    )
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train (Default: 100)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate (Default: 0.001)")
    return parser.parse_args()

def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc="Train", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix(loss=running_loss/(total+1e-6), acc=f"{(100.*correct/total):.2f}%")

def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Test", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix(loss=test_loss/(total+1e-6), acc=f"{(100.*correct/total):.2f}%")
    acc = 100. * correct / total
    return acc

def main():
    args = parse_args() # 解析参数
    os.makedirs("saved_models", exist_ok=True)
    device = get_device()
    
    print("Loading CIFAR-10 dataset...")
    # 使用 get_cifar10_loaders (它包含标准化)
    trainloader, testloader = get_cifar10_loaders(batch_size=128)
    
    # --- 加载所选模型 ---
    print(f"Initializing {args.model.upper()} model...")
    if args.model == 'resnet18':
        model = ResNet18().to(device)
    elif args.model == 'vgg16':
        model = VGG16_BN().to(device)
    elif args.model == 'densenet121':
        model = DenseNet121().to(device)
    else:
        raise ValueError("Unknown model")
    
    SAVE_PATH = f"saved_models/cifar10_{args.model}.pth"
    print(f"Model will be saved to: {SAVE_PATH}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train(model, trainloader, criterion, optimizer, device)
        acc = test(model, testloader, criterion, device)
        
        if acc > best_acc:
            print(f"New best accuracy: {acc:.2f}%. Saving model to {SAVE_PATH}...")
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"\nTraining finished. Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
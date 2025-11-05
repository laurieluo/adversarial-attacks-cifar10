import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import logging
import sys
from tqdm import tqdm

from src.models import ResNet18, VGG16_BN, DenseNet121
from src.data_loader import get_cifar10_loaders
from src.utils import get_device
from src.logger import setup_logging

def parse_args():
    """Parses command-line arguments."""
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
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Redirect tqdm to stderr
    progress_bar = tqdm(trainloader, desc="Train", leave=False, file=sys.stderr) 
    
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
        
        # Update postfix
        progress_bar.set_postfix(loss=running_loss/(total+1e-6), acc=f"{(100.*correct/total):.2f}%")

def test(model, testloader, criterion, device):
    """Evaluates the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # Redirect tqdm to stderr
        progress_bar = tqdm(testloader, desc="Test", leave=False, file=sys.stderr) 
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update postfix
            progress_bar.set_postfix(loss=test_loss/(total+1e-6), acc=f"{(100.*correct/total):.2f}%")
            
    acc = 100. * correct / total
    return acc

def main():
    setup_logging()  # 1. Initialize the logger
    args = parse_args()
    
    os.makedirs("saved_models", exist_ok=True)
    device = get_device()
    
    logging.info("Loading CIFAR-10 dataset...")
    # Use get_cifar10_loaders (it includes normalization)
    trainloader, testloader = get_cifar10_loaders(batch_size=128)
    
    # --- Load the selected model ---
    logging.info(f"Initializing {args.model.upper()} model...")
    if args.model == 'resnet18':
        model = ResNet18().to(device)
    elif args.model == 'vgg16':
        model = VGG16_BN().to(device)
    elif args.model == 'densenet121':
        model = DenseNet121().to(device)
    else:
        logging.error(f"Unknown model architecture: {args.model}")
        sys.exit(1)
    
    SAVE_PATH = f"saved_models/cifar10_{args.model}.pth"
    logging.info(f"Model will be saved to: {SAVE_PATH}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    logging.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Log epoch header (info level)
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train(model, trainloader, criterion, optimizer, device)
        acc = test(model, testloader, criterion, device)
        
        if acc > best_acc:
            logging.info(f"New best accuracy: {acc:.2f}%. Saving model to {SAVE_PATH}...")
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)

    logging.info(f"Training finished. Best test accuracy: {best_acc:.2f}%")
    logging.info(f"Final model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
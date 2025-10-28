# src/data_loader.py
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

# --- 1. Full CIFAR-10 Loaders (for Training) ---
def get_cifar10_loaders(batch_size=128):
    """
    Gets the full CIFAR-10 train and test DataLoaders.
    Applies standard augmentations and normalization.
    """
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./dataset', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root='./dataset', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# --- 2. Custom 500-Image Dataset (for Attacking) ---

class CustomCleanDataset(Dataset):
    """
    Loads your custom cifar10_clean_500 dataset.
    """
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = {}
        
        if not os.path.exists(label_file):
            print(f"Error: Label file not found at {label_file}")
            sys.exit(1)
            
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.labels[parts[0]] = int(parts[1])
        
        # Assumes 500 images named 0.png, 1.png, ..., 499.png
        self.image_files = [f"{i}.png" for i in range(500)]
        
        if not os.path.isdir(image_dir):
            print(f"Error: Image directory not found at {image_dir}")
            sys.exit(1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx] # e.g., "0.png"
        img_path = os.path.join(self.image_dir, img_name)
        
        if not os.path.exists(img_path):
            # Return a placeholder if an image is missing
            # NEW: Also return a placeholder name
            return torch.zeros(3, 32, 32), -1, "missing.png" 

        image = Image.open(img_path).convert('RGB')
        label = self.labels.get(img_name, -1) 

        if self.transform:
            image = self.transform(image)

        # NEW: Return the image name along with the image and label
        return image, label, img_name

def get_custom_loader(image_dir, label_file, batch_size=32):
    """
    Gets the DataLoader for your custom 500-image dataset.
    """
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),      
    ])
    
    dataset = CustomCleanDataset(
        image_dir=image_dir, 
        label_file=label_file, 
        transform=transform
    )
    
    # The default collate_fn will correctly batch tensors
    # and put the string filenames into a list/tuple.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
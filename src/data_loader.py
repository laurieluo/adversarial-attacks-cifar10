import os
import sys
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

# --- 1. Full CIFAR-10 Loaders (for Training) ---
def get_cifar10_loaders(batch_size=128):
    """
    Gets the full CIFAR-10 train and test DataLoaders.
    Applies standard augmentations and normalization.
    """
    logging.info("Initializing CIFAR-10 data loaders...")
    
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

    try:
        trainset = datasets.CIFAR10(
            root='./dataset', train=True, download=True, transform=transform_train)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(
            root='./dataset', train=False, download=True, transform=transform_test)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        logging.info("CIFAR-10 train/test loaders created successfully.")
        
    except Exception as e:
        logging.error(f"Failed to download or load CIFAR-10 dataset: {e}")
        logging.error("Please check your internet connection or dataset path permissions.")
        sys.exit(1)
    
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
        
        # 2. Use logging.error for fatal errors
        if not os.path.exists(label_file):
            logging.error(f"Label file not found at {label_file}")
            sys.exit(1)
            
        if not os.path.isdir(image_dir):
            logging.error(f"Image directory not found at {image_dir}")
            sys.exit(1)
        
        # 3. Use logging.info for success
        logging.info(f"Loading labels from: {label_file}")
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.labels[parts[0]] = int(parts[1])
        
        # Assumes 500 images named 0.png, 1.png, ..., 499.png
        self.image_files = [f"{i}.png" for i in range(500)]
        logging.info(f"Found {len(self.labels)} labels and {len(self.image_files)} image files.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx] # e.g., "0.png"
        img_path = os.path.join(self.image_dir, img_name)
        
        if not os.path.exists(img_path):
            # 4. Use logging.warning for non-fatal issues
            logging.warning(f"Image file not found: {img_path}. Returning placeholder.")
            # Return a placeholder
            return torch.zeros(3, 32, 32), -1, "missing.png"

        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels.get(img_name, -1) 
            if label == -1:
                logging.warning(f"No label found for image: {img_name}")

            if self.transform:
                image = self.transform(image)

            return image, label, img_name
        
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 32, 32), -1, "error.png"


def get_custom_loader(image_dir, label_file, batch_size=32):
    """
    Gets the DataLoader for your custom 500-image dataset.
    """
    logging.info("Initializing custom 500-image loader...")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),      
    ])
    
    dataset = CustomCleanDataset(
        image_dir=image_dir, 
        label_file=label_file, 
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    logging.info("Custom loader created successfully.")
    return dataloader
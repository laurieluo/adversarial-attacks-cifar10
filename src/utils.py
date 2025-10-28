# src/utils.py
import torch
import torch.nn as nn

def get_device():
    """
    Checks for and returns the best available device (MPS, CUDA, or CPU).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple (MPS) GPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

class NormalizedModel(nn.Module):
    """
    A wrapper class for models to automatically normalize input.
    
    This is necessary because libraries like torchattacks expect
    the model to handle normalization internally, while the attack
    is performed on images in the [0, 1] range.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # CIFAR-10 standard mean and std
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    def forward(self, x):
        # x is assumed to be in the [0, 1] range
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        
        # (x - mean) / std
        return self.model((x - self.mean) / self.std)
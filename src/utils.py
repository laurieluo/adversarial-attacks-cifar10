import torch
import torch.nn as nn
import logging
import shutil
import os

def get_device():
    """
    Checks for and returns the best available device (MPS, CUDA, or CPU).
    Logs the device being used.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple (MPS) GPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using NVIDIA (CUDA) GPU.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")
    return device

class NormalizedModel(nn.Module):
    """
    A wrapper class for models to automatically normalize input.
    
    This is necessary because libraries like torchattacks expect
    the model to handle normalization internally, while the attack
    is performed on images in the [0, 1] range.
    """
    def __init__(self, model):
        """
        :param model: The base model (e.g., ResNet18) to wrap.
        """
        super().__init__()
        self.model = model
        
        # CIFAR-10 standard mean and std
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

        # Log the normalization stats
        logging.info(f"NormalizedModel wrapper initialized.")
        logging.debug(f"Using Mean: {self.mean.view(-1).tolist()}")
        logging.debug(f"Using Std:  {self.std.view(-1).tolist()}")

    def forward(self, x):
        # x is assumed to be in the [0, 1] range
        
        # Ensure mean and std are on the same device as the input
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        
        # (x - mean) / std
        return self.model((x - self.mean) / self.std)

def create_zip_archive(archive_base_path, root_dir, base_dir="images"):
    """
    Creates a zip archive containing the specified directory.
    
    To get a zip file that unzips to 'images/...'
    - archive_base_path: 'path/to/output/archive_name' (no .zip)
    - root_dir: 'path/to/directory/containing/images'
    - base_dir: 'images'
    
    :param archive_base_path: The full path for the output zip, without the .zip extension.
    :param root_dir: The directory to 'cd' into before zipping (this becomes the root).
    :param base_dir: The directory *within* root_dir to zip up (e.g., "images").
    """
    try:
        # e.g., shutil.make_archive('.../PGD/archive', 'zip', '.../PGD', 'images')
        # This zips the 'images' folder found inside '.../PGD'
        # and saves it as '.../PGD/archive.zip'
        zip_path = shutil.make_archive(
            base_name=archive_base_path,
            format='zip',
            root_dir=root_dir,
            base_dir=base_dir
        )
        logging.info(f"Successfully created ZIP archive: {zip_path}")
    except Exception as e:
        logging.error(f"Failed to create ZIP archive at '{archive_base_path}.zip': {e}")
import torch
import torch.nn as nn
import logging
import shutil
import sys
import os
from datetime import datetime
from skimage.metrics import structural_similarity
from src.models import ResNet18, VGG16_BN, DenseNet121
from src.attacks import (
    PGD, FGSM, BIM, CW, AutoAttack, Pixle, 
    VNIFGSM, OnePixel, SparseFool, Jitter,
    PGD_CW
)


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

def load_model(model_name, device):
    """
    Loads a pre-trained model instance based on its name.
    """
    MODEL_PATH = f"saved_models/cifar10_{model_name.lower()}.pth"
    logging.info(f"Loading pre-trained {model_name.upper()} model from: {MODEL_PATH}")

    # 1. Instantiate the base model
    if model_name == 'resnet18':
        base_model = ResNet18().to(device)
    elif model_name == 'vgg16':
        base_model = VGG16_BN().to(device)
    elif model_name == 'densenet121':
        base_model = DenseNet121().to(device)
    else:
        logging.error(f"Invalid model name '{model_name}'.")
        sys.exit(1)

    # 2. Check for and load weights
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at '{MODEL_PATH}'")
        logging.error(f"Please run 'python train.py --model {model_name}' first.")
        sys.exit(1)
        
    try:
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    except RuntimeError as e:
        logging.error("Fatal: Weight loading failed (RuntimeError).")
        logging.error(e)
        logging.error(f"Architecture/weights mismatch for {model_name.upper()}.")
        sys.exit(1)
        
    base_model.eval()
    return base_model

def get_attack(attack_name, norm_model, batch_size):
    """
    Initializes and returns an attack instance based on its name.
    """
    if attack_name == 'pgd':
        atk = PGD(norm_model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    elif attack_name == 'fgsm':
        atk = FGSM(norm_model, eps=8/255)
    elif attack_name == 'bim':
        atk = BIM(norm_model, eps=8/255, alpha=2/255, steps=10)
    elif attack_name == 'cw':
        atk = CW(norm_model, c=1, kappa=0, steps=1000, lr=0.01)
    elif attack_name == 'autoattack':
        atk = AutoAttack(norm_model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
    elif attack_name == 'pixle':
        atk = Pixle(norm_model, x_dimensions=(2, 10), y_dimensions=(2, 10), pixel_mapping='random', restarts=20, max_iterations=10)
    elif attack_name == 'vnifgsm':
        atk = VNIFGSM(
            model=norm_model,
            eps=8/255,      # CIFAR-10 标准
            alpha=2/255,    # CIFAR-10 标准
            steps=10,       # CIFAR-10 标准
            decay=1.0,      # VNI 官方值
            n=20,           # VNI 官方值 (之前是 5)
            beta=1.5        # VNI 官方值
        )
    elif attack_name == 'onepixel':
        # OnePixel needs inf_batch
        atk = OnePixel(norm_model, pixels=1, steps=10, popsize=10, inf_batch=batch_size)
    elif attack_name == 'sparsefool':
        atk = SparseFool(norm_model, steps=10, lam=3, overshoot=0.02)
    elif attack_name == 'jitter':
        atk = Jitter(norm_model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)
    elif attack_name == 'pgd_cw':  # 新增：PGDCW混合攻击
        atk = PGD_CW(model=norm_model,pgd_eps=6/255,pgd_alpha=2/255,pgd_steps=10,pgd_random_start=True,cw_c=0.3,cw_kappa=0,cw_steps=300,cw_lr=0.01)
    else:
        logging.error(f"Unknown attack '{attack_name}'")
        sys.exit(1)
        
    return atk

def calculate_batch_ssim(clean_images_batch, adv_images_batch):
    """
    Calculates the total SSIM score for a batch of images.
    """
    clean_images_np = clean_images_batch.cpu().detach().numpy().transpose(0, 2, 3, 1)
    adv_images_np = adv_images_batch.cpu().detach().numpy().transpose(0, 2, 3, 1)
    
    batch_ssim_sum = 0.0
    for i in range(clean_images_np.shape[0]):
        ssim_score = structural_similarity(
            clean_images_np[i],
            adv_images_np[i],
            data_range=1.0,
            channel_axis=-1
        )
        batch_ssim_sum += ssim_score # type: ignore
        
    return batch_ssim_sum

def generate_results_table(attack_name, model_name, total_images, acc_clean, acc_adv, score_asr, score_ssim, score_m):
    """
    Formats the final results into a printable ASCII table.
    """
    attack_title = f" ATTACK: {attack_name.upper()} | MODEL: {model_name.upper()} "
    total_img_str = f"{total_images}"
    clean_acc_str = f"{acc_clean:.2f}%"
    adv_acc_str = f"{acc_adv:.2f}%"
    asr_str = f"{score_asr:.4f} ({(score_asr*100):.2f}%)"
    ssim_str = f"{score_ssim:.4f} ({(score_ssim*100):.2f}%)"
    m_score_str = f"{score_m:.4f}"

    max_width = 59
    col1_width = 32
    col2_width = 20
    inner_width = max_width - 2

    title_padding = (inner_width - len(attack_title)) // 2
    title_padding_rem = inner_width - len(attack_title) - title_padding
    
    if title_padding < 0: # Handle long titles
        max_width = len(attack_title) + 4
        inner_width = max_width - 2
        col1_width = 32
        col2_width = max_width - col1_width - 7
        title_padding = 1
        title_padding_rem = 1

    report_table = "\n" 
    report_table += f"+{'=' * (max_width - 2)}+\n"
    report_table += f"|{' ' * title_padding}{attack_title}{' ' * title_padding_rem}|\n"
    report_table += f"+{'=' * (max_width - 2)}+\n"
    report_table += f"| {'Metric':<{col1_width}} | {'Value':<{col2_width}} |\n"
    report_table += f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|\n"
    report_table += f"| {'Total Images Tested':<{col1_width}} | {total_img_str:<{col2_width}} |\n"
    report_table += f"| {'Clean Accuracy':<{col1_width}} | {clean_acc_str:<{col2_width}} |\n"
    report_table += f"| {'Adversarial Accuracy':<{col1_width}} | {adv_acc_str:<{col2_width}} |\n"
    report_table += f"| {'Attack Success Rate (ASR)':<{col1_width}} | {asr_str:<{col2_width}} |\n"
    report_table += f"| {'Avg. Structural Sim. (SSIM)':<{col1_width}} | {ssim_str:<{col2_width}} |\n"
    report_table += f"| {'Composite Score (M)':<{col1_width}} | {m_score_str:<{col2_width}} |\n"
    report_table += f"+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+\n"
    
    return report_table

def save_and_archive_results(attack_output_dir, label_file, model_name, attack_name):
    """
    Copies the label file and creates a ZIP archive of the results.
    """
    try:
        label_save_path = os.path.join(attack_output_dir, "label.txt")
        shutil.copy(label_file, label_save_path)
        logging.info(f"Copied label file to: {label_save_path}")
    except Exception as e:
        logging.warning(f"Could not copy label file: {e}")

    logging.info("Creating ZIP archive...")
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_file_name = f"{model_name.upper()}_{attack_name.upper()}_{timestamp_str}"
    archive_base_path = os.path.join(attack_output_dir, zip_file_name)
    
    # Assuming create_zip_archive is defined elsewhere in utils.py
    from src.utils import create_zip_archive # Or ensure it's in scope
    create_zip_archive(
        archive_base_path=archive_base_path,
        root_dir=attack_output_dir,
        base_dir="images" 
    )

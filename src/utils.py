import torch
import torch.nn as nn
import logging
import shutil
import sys
import os
import re
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity
from src.models import ResNet18, VGG16_BN, DenseNet121
from src.wideresnet import WideResNet28_10, WideResNet94_16, WideResNet70_16
# Attack imports are moved to get_attack() function to avoid import errors
# when TransferAttack is not available (e.g., in test_asr.py)


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
        normalizes_internally = (
            isinstance(getattr(model, "mean", None), torch.Tensor)
            and isinstance(getattr(model, "std", None), torch.Tensor)
        )
        self.should_normalize = not normalizes_internally

        if self.should_normalize:
            # CIFAR-10 standard mean and std
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

            logging.info("NormalizedModel wrapper initialized with CIFAR-10 statistics.")
            logging.debug(f"Using Mean: {self.mean.view(-1).tolist()}")
            logging.debug(f"Using Std:  {self.std.view(-1).tolist()}")
        else:
            self.mean = None
            self.std = None
            logging.info("NormalizedModel wrapper detected internal normalization; skipping external normalization.")

    def forward(self, x):
        # x is assumed to be in the [0, 1] range
        if not self.should_normalize:
            return self.model(x)
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return self.model((x - mean) / std)

class EnsembleModel(nn.Module):
    def __init__(self, models):
        """
        集成模型包装类
        :param models: 模型列表
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)  # 使用 ModuleList 管理子模型
        self.num_models = len(models)

    def forward(self, x):
        """
        前向传播 - 返回模型输出的平均值
        """
        outputs = [model(x) for model in self.models]
        return sum(outputs) / len(outputs)

def create_ensemble_model(model_names, device):
    """
    创建集成模型
    :param model_names: 模型名称列表
    :param device: 设备
    :return: EnsembleModel 实例
    """
    models = []
    for name in model_names:
        base_model = load_model(name, device)
        norm_model = NormalizedModel(base_model).to(device)
        norm_model.eval()
        models.append(norm_model)

    # 创建继承自 nn.Module 的集成模型，确保 num_model 与实际模型数一致
    class EnsembleModelWrapper(nn.Module):
        def __init__(self, model_list):
            super(EnsembleModelWrapper, self).__init__()
            self.models = nn.ModuleList(model_list)
            self.num_model = len(model_list)  # 确保数量一致

        def forward(self, x):
            # 集成模型前向传播（这里简单平均）
            outputs = [model(x) for model in self.models]
            return sum(outputs) / len(outputs)

    return EnsembleModelWrapper(models)

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

# 特征层映射字典
MODEL_FEATURE_LAYERS = {
    'resnet18': 'layer2',  # ResNet18 中间残差块
    'vgg16': 'features.15',  # VGG16 第3卷积块的最后一层
    'densenet121': 'features.denseblock2',  # DenseNet121 第2个密集块
    'wrn': 'layer.1'    # WRN28-10 和 WRN94-16 中间残差块组
}

def validate_feature_layer(model, layer_name):
    """验证模型中是否存在指定的特征层"""
    for name, _ in model.named_modules():
        if name == layer_name:
            return True
    return False

def load_model(model_name, device):
    """
    Loads a pre-trained model instance based on its name.
    """
    normalized_name = re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')

    MODEL_REGISTRY = {
        'resnet18': {'factory': ResNet18, 'weight': 'cifar10_resnet18.pth'},
        'vgg16': {'factory': VGG16_BN, 'weight': 'cifar10_vgg16.pth'},
        'densenet121': {'factory': DenseNet121, 'weight': 'cifar10_densenet121.pth'},
        'wrn2810': {'factory': WideResNet28_10, 'weight': 'Cui2023Decoupled_wrn-28-10.pt'},
        'wrn9416': {'factory': WideResNet94_16, 'weight': 'Bartoldson2024Adversarial_WRN-94-16.pt'},
        'wrn7016': {'factory': WideResNet70_16, 'weight': 'Wang2023Better_wrn-70-16.pt'},
    }

    MODEL_ALIASES = {
        'resnet18': 'resnet18',
        'vgg16': 'vgg16',
        'densenet121': 'densenet121',
        'wrn2810': 'wrn2810',
        'wrn9416': 'wrn9416',
        'wrn7016': 'wrn7016',
        'wang2023better_wrn_70_16': 'wrn7016',  # Support full name format
        'wang2023better': 'wrn7016',  # Support shortened name
    }

    canonical_name = MODEL_ALIASES.get(normalized_name, normalized_name)
    if canonical_name not in MODEL_REGISTRY:
        logging.error(f"Invalid model name '{model_name}'.")
        sys.exit(1)

    spec = MODEL_REGISTRY[canonical_name]
    model_factory = spec['factory']
    MODEL_PATH = os.path.join("saved_models", spec['weight'])
    logging.info(f"Loading pre-trained {model_name.upper()} model from: {MODEL_PATH}")

    # 1. Instantiate the base model
    base_model = model_factory().to(device)

    # 2. Verify the feature layer
    feature_layer = MODEL_FEATURE_LAYERS.get(canonical_name)
    if feature_layer and not validate_feature_layer(base_model, feature_layer):
        logging.warning(f"Feature layer '{feature_layer}' not found in {model_name}")
        sys.exit(1)
    else:
        logging.info(f"Using feature layer '{feature_layer}' for {model_name.upper()}")

    # 3. Check for and load weights
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

def get_attack(attack_name, norm_model, batch_size, device, surrogate_name=None):
    """
    Initializes and returns an attack instance based on its name.
    """
    # Lazy import of attacks to avoid import errors when TransferAttack is not available

    # 1. 从norm_model中获取基础模型的类名
    base_model_class_name = norm_model.model.__class__.__name__

    # 2. 建立类名与标准模型名称的映射（与load_model中的逻辑对应）
    model_class_to_name = {
        "ResNet": "resnet18",
        "VGG": "vgg16",
        "DenseNet": "densenet121",
        "DMWideResNet": "wrn"
    }

    # 3. 解析标准模型名称
    try:
        model_name = model_class_to_name[base_model_class_name]
        logging.info(f"解析到标准模型名称: {model_name}")
    except KeyError:
        logging.error(f"未知的模型类: {base_model_class_name}，无法解析标准模型名称")
        sys.exit(1)

    from src.attacks import (
        PGD, FGSM, BIM, CW, AutoAttack, Pixle, 
        VNIFGSM, OnePixel, SparseFool, Jitter,
        PGD_CW, VNIFGSM_SIM, Pixle_VNIFGSM, AIFGTM,
        AdaEA, CWA, OPS, L2T, RFAInf, P2FA, BFA
    )
    
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
        atk = Pixle(norm_model, x_dimensions=(10, 20), y_dimensions=(10, 20), pixel_mapping='random', restarts=20, max_iterations=10)
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
        atk = OnePixel(norm_model, pixels=1, steps=10, popsize=10, inf_batch=batch_size)
    elif attack_name == 'sparsefool':
        atk = SparseFool(norm_model, steps=10, lam=3, overshoot=0.02)
    elif attack_name == 'jitter':
        atk = Jitter(norm_model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)
    elif attack_name == 'pgd_cw':
        atk = PGD_CW(model=norm_model,pgd_eps=6/255,pgd_alpha=2/255,pgd_steps=10,pgd_random_start=True,cw_c=0.3,cw_kappa=0,cw_steps=300,cw_lr=0.01)
    elif attack_name == 'vnifgsm_sim':
        atk = VNIFGSM_SIM(model=norm_model, eps=8/255, alpha=2/255, steps=10, decay=1.0, n=20, beta=1.5, num_scale=5, scale_factor=1.1, momentum_weight=0.6, sim_weight=0.4)
    elif attack_name == 'pixle_vnifgsm':
        atk = Pixle_VNIFGSM(model=norm_model,eps=8/255,alpha=2/255,steps=15)
    elif attack_name == 'aifgtm':
        atk = AIFGTM(model=norm_model, eps=8/255, alpha=2/255, steps=10, decay=1.0, beta_1=0.9, beta_2=0.99, lam=1.3, mu_1=1.5, mu_2=1.9)
    elif attack_name == 'adaea':
        atk = AdaEA(model=norm_model, eps=16/255, alpha=1.6/255, steps=10, decay=1.0, beta=10, threshold=-0.3)
    elif attack_name == 'cwa':
        atk = CWA(model=norm_model, eps=16/255, alpha=3.2/255, steps=10, decay=1.0, beta=50, r_size=16/255/15, inner_step_size=250)

    elif attack_name == 'ops':
        # OPS optimized for CIFAR-10 (32x32)
        # Using smaller eps (8/255) and reduced num_sample_operator for better performance on small images
        atk = OPS(
            model=norm_model,
            eps=16/255,       # Reduced from 16/255 for CIFAR-10 (better balance)
            alpha=1.6/255,     # Will be set to eps/steps
            steps=10,       # Official default
            decay=1.0,       # Official default
            beta=2.,         # Official default
            num_sample_neighbor=30,    # Official default
            num_sample_operator=30,    # Reduced from 20 for CIFAR-10 (faster, similar effectiveness)
            sample_levels=range(1, 5),
            sample_ratios=np.arange(0., 3, 0.5) + 0.5
        )

    elif attack_name == 'l2t':
        # L2T optimized for CIFAR-10 (32x32)
        # Using smaller eps (8/255) for better performance on small images
        atk = L2T(
            model=norm_model,
            eps=8/255,       # Reduced from 16/255 for CIFAR-10 (better balance)
            alpha=None,     # Will be set to eps/steps
            steps=10,       # Official default
            decay=1.0,       # Official default
            num_scale=3     # Official default
        )

    elif attack_name == 'rfa_inf':
        if surrogate_name:
            logging.info(f"Loading RFA∞ surrogate model '{surrogate_name}'.")
            surrogate_base = load_model(surrogate_name, device)
            surrogate_model = NormalizedModel(surrogate_base).to(device)
            surrogate_model.eval()
        else:
            logging.warning("No surrogate model provided; using victim model for gradients.")
            surrogate_model = norm_model

        atk = RFAInf(
            model=norm_model,
            surrogate_model=surrogate_model,
            eps=0.1,
            alpha=2 / 255,
            steps=50,
            random_start=True,
        )

    elif attack_name == 'p2fa':
        atk = P2FA(
            model=norm_model,
            eps=16/255,
            alpha=1.6/255,
            steps=10,
            decay=1.0,
            num_ens=30,
            feature_layer=MODEL_FEATURE_LAYERS[model_name],
            eta=28.0
        )

    elif attack_name == 'bfa':
        atk = BFA(
            model=norm_model,
            eps=16/255,
            alpha=1.6/255,
            steps=10,
            decay=1.0,
            num_ens=30,
            layer_name=MODEL_FEATURE_LAYERS[model_name],
            eta=28.0
        )

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

def load_adversarial_images(adv_image_dir, label_file, batch_size=32):
    """
    Loads adversarial images from a directory.
    
    :param adv_image_dir: Directory containing adversarial images
    :param label_file: Path to label file (same format as clean dataset)
    :param batch_size: Batch size for DataLoader
    :return: DataLoader for adversarial images
    """
    from src.data_loader import CustomCleanDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    dataset = CustomCleanDataset(
        image_dir=adv_image_dir,
        label_file=label_file,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logging.info(f"Loaded adversarial images from: {adv_image_dir}")
    return dataloader

def calculate_asr(model, adv_loader, device):
    """
    Calculates Attack Success Rate (ASR) on adversarial images.
    
    ASR = (Number of successful attacks) / (Total number of images)
    A successful attack means the model misclassifies the adversarial image.
    
    :param model: The model to evaluate (should be wrapped with NormalizedModel)
    :param adv_loader: DataLoader containing adversarial images and labels
    :param device: Device to run evaluation on
    :return: Dictionary containing ASR, accuracy, total_images, and successful_attacks
    """
    model.eval()
    total_correct = 0
    total_images = 0
    
    with torch.no_grad():
        for images, labels, img_names in adv_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter out invalid labels
            valid_idx_bool = (labels != -1)
            if not valid_idx_bool.any():
                continue
            
            images_batch = images[valid_idx_bool]
            labels_batch = labels[valid_idx_bool]
            
            # Get predictions
            outputs = model(images_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            # Count correct predictions
            correct = (predicted == labels_batch).sum().item()
            total_correct += correct
            total_images += labels_batch.size(0)
    
    if total_images == 0:
        logging.warning("No valid images found for ASR calculation.")
        return {
            'asr': 0.0,
            'accuracy': 0.0,
            'total_images': 0,
            'successful_attacks': 0
        }
    
    accuracy = 100.0 * total_correct / total_images
    successful_attacks = total_images - total_correct
    asr = successful_attacks / total_images
    
    return {
        'asr': asr,
        'accuracy': accuracy,
        'total_images': total_images,
        'successful_attacks': successful_attacks
    }

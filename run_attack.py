import torch
import os
import sys
import argparse
import shutil
import logging
import numpy as np
from datetime import datetime
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity

from src.logger import setup_logging
from src.models import ResNet18, VGG16_BN, DenseNet121
from src.data_loader import get_custom_loader
from src.utils import get_device, NormalizedModel, create_zip_archive
from src.attacks import PGD, FGSM, BIM, CW, AutoAttack, Pixle, VNIFGSM, OnePixel, SparseFool, Jitter

# --- Configuration ---
IMAGE_DIR = "dataset/cifar10_clean_500/images"
LABEL_FILE = "dataset/cifar10_clean_500/label.txt"
OUTPUT_DIR = "adversarial_images"
BATCH_SIZE = 32

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on CIFAR-10")
    parser.add_argument(
        '--attack',
        type=str,
        default='pgd',
        choices=['pgd', 'fgsm', 'bim', 'cw', 'autoattack', 'pixle', 'vnifgsm',
                 'onepixel', 'sparsefool', 'jitter'
                ],
        help="Type of attack to run (default: pgd)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        choices=['resnet18', 'vgg16', 'densenet121'],
        help="Model architecture to attack (default: resnet18)"
    )
    parser.add_argument(
        '--save_images',
        action='store_true',
        help="Save the generated adversarial images to disk"
    )
    return parser.parse_args()

# --- Main Execution ---
def main():
    setup_logging()
    args = parse_args()
    logging.info(f"Running Attack: {args.attack.upper()} on Model: {args.model.upper()}")

    # Define attack-specific output paths including the model name
    attack_output_dir = os.path.join(OUTPUT_DIR, args.model.upper(), args.attack.upper())
    images_save_dir = os.path.join(attack_output_dir, "images")
    
    # Only create directories and print if saving
    if args.save_images:
        os.makedirs(images_save_dir, exist_ok=True)
        logging.info(f"Adversarial images will be saved to: {images_save_dir}")
    else:
        logging.warning("Note: Adversarial images will NOT be saved (use --save_images to save).")
    
    device = get_device()
    
    # Dynamically set model path and load model instance
    model_name = args.model.lower()
    MODEL_PATH = f"saved_models/cifar10_{model_name}.pth"
    
    logging.info(f"Loading pre-trained {model_name.upper()} model from: {MODEL_PATH}")
    
    if model_name == 'resnet18':
        base_model = ResNet18().to(device)
    elif model_name == 'vgg16':
        base_model = VGG16_BN().to(device)
    elif model_name == 'densenet121':
        base_model = DenseNet121().to(device)
    else:
        logging.error(f"Invalid model name '{model_name}'. This should not happen.")
        sys.exit(1)

    # Load the *base* pre-trained model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at '{MODEL_PATH}'")
        logging.error(f"Please run 'python train.py --model {model_name}' first to train and save the model.")
        sys.exit(1)
        
    try:
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    except RuntimeError as e:
        logging.error("--- Fatal: Weight loading failed (RuntimeError) ---")
        logging.error(e)
        logging.error(f"This usually means the architecture in 'src/models.py' for {model_name.upper()} does not match the '.pth' file.")
        sys.exit(1)
        
    base_model.eval()

    # Wrap the model for normalization
    norm_model = NormalizedModel(base_model).to(device)
    norm_model.eval()

    # Load CIFAR10 500-image dataset
    logging.info(f"Loading custom dataset from: {IMAGE_DIR}")
    custom_loader = get_custom_loader(
        image_dir=IMAGE_DIR,
        label_file=LABEL_FILE,
        batch_size=BATCH_SIZE
    )

    # --- Initialize the selected attack ---
    if args.attack == 'pgd':
        atk = PGD(norm_model, eps=8/255, alpha=2/255, steps=10, random_start=True)

    elif args.attack == 'fgsm':
        atk = FGSM(norm_model, eps=8/255)

    elif args.attack == 'bim':
        atk = BIM(norm_model, eps=8/255, alpha=2/255, steps=10)

    elif args.attack == 'cw':
        atk = CW(norm_model, c=1, kappa=0, steps=1000, lr=0.01)

    elif args.attack == 'autoattack':
        atk = AutoAttack(norm_model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)

    elif args.attack == 'pixle':
        atk = Pixle(norm_model, x_dimensions=(2, 10), y_dimensions=(2, 10), pixel_mapping='random', restarts=20, max_iterations=10)

    elif args.attack == 'vnifgsm':
        atk = VNIFGSM(norm_model, eps=8/255, alpha=2/255, steps=10, decay=1.0, n=5, beta=1.5)

    elif args.attack == 'onepixel':
        atk = OnePixel(norm_model, pixels=1, steps=10, popsize=10, inf_batch=BATCH_SIZE)

    elif args.attack == 'sparsefool':
        atk = SparseFool(norm_model, steps=10, lam=3, overshoot=0.02)

    elif args.attack == 'jitter':
        atk = Jitter(norm_model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)

    else:
        logging.error(f"Unknown attack '{args.attack}'")
        sys.exit(1)

    # Run evaluation and attack
    total_correct_clean = 0
    total_correct_adv = 0
    total_images = 0
    total_ssim_sum = 0.0

    # Redirect tqdm output to sys.stderr to avoid conflict with logs on stdout
    progress_bar = tqdm(custom_loader, desc=f"Attacking ({args.attack.upper()})", leave=True, file=sys.stderr)
    
    for images, labels, img_names in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        valid_idx_bool = (labels != -1)
        if not valid_idx_bool.any():
            continue
        
        images_clean_batch = images[valid_idx_bool]
        labels_batch = labels[valid_idx_bool]
        img_names_batch = [name for i, name in enumerate(img_names) if valid_idx_bool[i]]

        # --- Generate adversarial images ---
        adv_images = atk.attack(images_clean_batch, labels_batch)

        # --- Calculate SSIM Score ---
        clean_images_np = images_clean_batch.cpu().detach().numpy().transpose(0, 2, 3, 1)
        adv_images_np = adv_images.cpu().detach().numpy().transpose(0, 2, 3, 1)

        batch_ssim_sum = 0.0
        for i in range(clean_images_np.shape[0]):
            ssim_score = structural_similarity(
                clean_images_np[i],
                adv_images_np[i],
                data_range=1.0,
                channel_axis=-1
            )
            batch_ssim_sum += ssim_score # type: ignore
        
        total_ssim_sum += batch_ssim_sum

        # Test on clean images
        with torch.no_grad():
            outputs_clean = norm_model(images_clean_batch)
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            total_correct_clean += (predicted_clean == labels_batch).sum().item()

        # Test on adversarial images
        with torch.no_grad():
            outputs_adv = norm_model(adv_images)
            _, predicted_adv = torch.max(outputs_adv.data, 1)
            total_correct_adv += (predicted_adv == labels_batch).sum().item()

        total_images += labels_batch.size(0)

        if args.save_images:
            for i in range(len(adv_images)):
                adv_img_tensor = adv_images[i]
                img_name = img_names_batch[i] # e.g., "42.png"
                save_path = os.path.join(images_save_dir, img_name)
                save_image(adv_img_tensor, save_path)
                
        progress_bar.set_postfix(clean_acc=f"{(100.*total_correct_clean/total_images):.2f}%",
                                 adv_acc=f"{(100.*total_correct_adv/total_images):.2f}%")

    # --- Print final report with scores ---
    if total_images > 0:
        acc_clean = 100. * total_correct_clean / total_images
        acc_adv = 100. * total_correct_adv / total_images

        score_asr = (total_images - total_correct_adv) / total_images
        score_ssim = total_ssim_sum / total_images
        score_m = 100 * score_asr * score_ssim

        # --- Format the results as a table ---
        
        attack_title = f" ATTACK: {args.attack.upper()} | MODEL: {args.model.upper()} "
        total_img_str = f"{total_images}"
        clean_acc_str = f"{acc_clean:.2f}%"
        adv_acc_str = f"{acc_adv:.2f}%"
        asr_str = f"{score_asr:.4f} ({(score_asr*100):.2f}%)"
        ssim_str = f"{score_ssim:.4f} ({(score_ssim*100):.2f}%)"
        m_score_str = f"{score_m:.4f}"

        # Calculate widths - Increased for better readability
        max_width = 59 # Increased from 46
        col1_width = 32 # Increased from 24
        col2_width = 20 # Increased from 17
        inner_width = max_width - 2

        # Adjust title padding dynamically
        title_padding = (inner_width - len(attack_title)) // 2
        title_padding_rem = inner_width - len(attack_title) - title_padding

        # Handle cases where title is too long (adjust max_width if necessary)
        if title_padding < 0:
            max_width = len(attack_title) + 4 # Add 4 for " | " and padding
            inner_width = max_width - 2
            col1_width = 32 # Keep col1 fixed
            col2_width = max_width - col1_width - 7 # Adjust col2
            title_padding = 1
            title_padding_rem = 1
            
        
        # Build the table string
        report_table = "\n"  # Start with a newline for spacing
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
        
        # Adjust bottom line to new max_width
        report_table += f"+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+\n"


        # Log the entire table as a single info message
        logging.info(report_table)

        if args.save_images:
            try:
                label_save_path = os.path.join(attack_output_dir, "label.txt")
                shutil.copy(LABEL_FILE, label_save_path)
                logging.info(f"Copied label file to: {label_save_path}")
            except Exception as e:
                logging.warning(f"Could not copy label file: {e}")

            logging.info("Creating ZIP archive...")
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_file_name = f"{args.model.upper()}_{args.attack.upper()}_{timestamp_str}"
            archive_base_path = os.path.join(attack_output_dir, zip_file_name)
            create_zip_archive(
                archive_base_path=archive_base_path,
                root_dir=attack_output_dir,
                base_dir="images" 
            )

    else:
        logging.error("No valid images were processed. Check dataset paths.")

if __name__ == "__main__":
    main()
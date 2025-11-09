# run_attack.py
from src.logger import setup_logging
import logging
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
transfer_attack_path = os.path.join(project_root, 'TransferAttack')

if not os.path.exists(transfer_attack_path):
    logging.error(f"'TransferAttack' directory not found in {transfer_attack_path}.")
    logging.warning("Please make sure you run the command at project root: git clone https://github.com/Trustworthy-AI-Group/TransferAttack.git")
else:
    sys.path.insert(0, transfer_attack_path)


import torch
import argparse
from torchvision.utils import save_image
from tqdm import tqdm

from src.data_loader import get_custom_loader
from src.utils import (
    get_device, 
    NormalizedModel, 
    load_model, 
    get_attack, 
    calculate_batch_ssim,
    generate_results_table,
    save_and_archive_results
)

# --- Configuration ---
IMAGE_DIR = "dataset/cifar10_clean_500/images"
LABEL_FILE = "dataset/cifar10_clean_500/label.txt"
OUTPUT_DIR = "adversarial_images"
BATCH_SIZE = 32

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on CIFAR-10")
    parser.add_argument(
        '--attack', type=str, default='pgd',
        choices=['pgd', 'fgsm', 'bim', 'cw', 'autoattack', 'pixle', 'vnifgsm',
                 'onepixel', 'sparsefool', 'jitter', 'pgd_cw', 'vnifgsm_sim', 'pixle_vnifgsm', 'aifgtm'],
        help="Type of attack to run (default: pgd)"
    )
    parser.add_argument(
        '--model', type=str, default='resnet18',
        choices=['resnet18', 'vgg16', 'densenet121'],
        help="Model architecture to attack (default: resnet18)"
    )
    parser.add_argument(
        '--save-images', action='store_true',
        help="Save the generated adversarial images to disk"
    )
    return parser.parse_args()

# --- Main Execution ---
def main():
    setup_logging()
    args = parse_args()
    logging.info(f"Running Attack: {args.attack.upper()} on Model: {args.model.upper()}")

    # Define paths
    attack_output_dir = os.path.join(OUTPUT_DIR, args.model.upper(), args.attack.upper())
    images_save_dir = os.path.join(attack_output_dir, "images")
    
    if args.save_images:
        os.makedirs(images_save_dir, exist_ok=True)
        logging.info(f"Adversarial images will be saved to: {images_save_dir}")
    else:
        logging.warning("Note: Adversarial images will NOT be saved (use --save-images to save).")
    
    device = get_device()
    
    # --- 1. Load Model ---
    base_model = load_model(args.model, device)
    norm_model = NormalizedModel(base_model).to(device)
    norm_model.eval()

    # --- 2. Load Data ---
    logging.info(f"Loading custom dataset from: {IMAGE_DIR}")
    custom_loader = get_custom_loader(
        image_dir=IMAGE_DIR,
        label_file=LABEL_FILE,
        batch_size=BATCH_SIZE
    )

    # --- 3. Initialize Attack ---
    atk = get_attack(args.attack, norm_model, BATCH_SIZE)

    # --- 4. Run Evaluation and Attack ---
    total_correct_clean = 0
    total_correct_adv = 0
    total_images = 0
    total_ssim_sum = 0.0

    progress_bar = tqdm(custom_loader, desc=f"Attacking ({args.attack.upper()})", leave=True, file=sys.stderr)
    
    for images, labels, img_names in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        valid_idx_bool = (labels != -1)
        if not valid_idx_bool.any():
            continue
        
        images_clean_batch = images[valid_idx_bool]
        labels_batch = labels[valid_idx_bool]
        img_names_batch = [name for i, name in enumerate(img_names) if valid_idx_bool[i]]

        # Generate adversarial images
        adv_images = atk.attack(images_clean_batch, labels_batch)

        # --- 4a. Calculate SSIM ---
        total_ssim_sum += calculate_batch_ssim(images_clean_batch, adv_images)

        # --- 4b. Test Clean Images ---
        with torch.no_grad():
            outputs_clean = norm_model(images_clean_batch)
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            total_correct_clean += (predicted_clean == labels_batch).sum().item()

        # --- 4c. Test Adversarial Images ---
        with torch.no_grad():
            outputs_adv = norm_model(adv_images)
            _, predicted_adv = torch.max(outputs_adv.data, 1)
            total_correct_adv += (predicted_adv == labels_batch).sum().item()

        total_images += labels_batch.size(0)

        # --- 4d. Save Images ---
        if args.save_images:
            for i in range(len(adv_images)):
                save_path = os.path.join(images_save_dir, img_names_batch[i])
                save_image(adv_images[i], save_path)
                
        progress_bar.set_postfix(clean_acc=f"{(100.*total_correct_clean/total_images):.2f}%",
                                     adv_acc=f"{(100.*total_correct_adv/total_images):.2f}%")

    # --- 5. Print Final Report ---
    if total_images > 0:
        acc_clean = 100. * total_correct_clean / total_images
        acc_adv = 100. * total_correct_adv / total_images
        score_asr = (total_images - total_correct_adv) / total_images
        score_ssim = total_ssim_sum / total_images
        score_m = 100 * score_asr * score_ssim

        # --- 5a. Generate Report Table ---
        report_table = generate_results_table(
            args.attack, args.model, total_images, 
            acc_clean, acc_adv, score_asr, score_ssim, score_m
        )
        logging.info(report_table)

        # --- 5b. Archive Results ---
        if args.save_images:
            save_and_archive_results(
                attack_output_dir=attack_output_dir,
                label_file=LABEL_FILE,
                model_name=args.model,
                attack_name=args.attack
            )
    else:
        logging.error("No valid images were processed. Check dataset paths.")

if __name__ == "__main__":
    main()

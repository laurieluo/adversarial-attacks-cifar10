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
    create_ensemble_model,
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
                 'onepixel', 'sparsefool', 'jitter', 'pgd_cw', 'vnifgsm_sim',
                 'pixle_vnifgsm', 'aifgtm', 'adaea', 'cwa', 'ops', 'l2t', 'rfa_inf', 'p2fa'],
        help="Type of attack to run (default: pgd)"
    )
    parser.add_argument(
        '--model', type=str, default='resnet18',
        choices=['resnet18', 'vgg16', 'densenet121', 'wrn2810', 'wrn9416'],
        help="Model architecture to attack (default: resnet18)"
    )
    parser.add_argument(
        '--target-model', type=str, default=None,
        choices=['resnet18', 'vgg16', 'densenet121', 'wrn2810', 'wrn9416'],
        help="Victim model when using rfa_inf (optional otherwise)."
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

    if args.attack == 'rfa_inf':
        surrogate_model_name = args.model
        if args.target_model:
            target_model_name = args.target_model
        else:
            target_model_name = 'resnet18'
            logging.info("No target model provided for RFA∞; defaulting to RESNET18.")
    else:
        surrogate_model_name = None
        target_model_name = args.target_model or args.model
        if args.target_model and args.target_model != args.model:
            logging.warning("--target-model is only used by rfa_inf; ignoring for other attacks.")

    logging.info(f"Running Attack: {args.attack.upper()} on Model: {target_model_name.upper()}")
    if surrogate_model_name:
        logging.info(f"Using surrogate model: {surrogate_model_name.upper()}")

    # Define paths
    attack_output_dir = os.path.join(OUTPUT_DIR, target_model_name.upper(), args.attack.upper())
    images_save_dir = os.path.join(attack_output_dir, "images")
    
    if args.save_images:
        os.makedirs(images_save_dir, exist_ok=True)
        logging.info(f"Adversarial images will be saved to: {images_save_dir}")
    else:
        logging.warning("Note: Adversarial images will NOT be saved (use --save-images to save).")
    
    device = get_device()
    
    # --- 1. Load Model ---
    ENSEMBLE_ATTACKS = ['adaea', 'cwa']

    if args.attack in ENSEMBLE_ATTACKS:
        # 为集成攻击创建集成模型，确保只使用实际存在的模型
        ensemble_models = ['resnet18', 'vgg16', 'densenet121', 'wrn2810', 'wrn9416']  # 确保这些模型都存在
        # 检查模型文件是否存在
        available_models = []
        for model_name in ensemble_models:
            lower_name = model_name.lower()

            # 根据模型名称选择模型路径
            if lower_name == 'wrn2810':
                model_path = "saved_models/Cui2023Decoupled_wrn-28-10.pt"
            elif lower_name == 'wrn9416':
                model_path = "saved_models/Bartoldson2024Adversarial_WRN-94-16.pt"
            else:
                model_path = f"saved_models/cifar10_{lower_name}.pth"

            if os.path.exists(model_path):
                available_models.append(model_name)
            else:
                logging.warning(f"Model {model_name} not found at {model_path}, skipping.")

        if len(available_models) == 0:
            logging.error("No models available for AdaEA attack!")
            sys.exit(1)

        norm_model = create_ensemble_model(available_models, device)
        logging.info(f"Using {len(available_models)} models for AdaEA: {available_models}")
    else:
        # 单模型情况
        base_model = load_model(target_model_name, device)
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
    atk = get_attack(
        attack_name=args.attack,
        norm_model=norm_model,
        batch_size=BATCH_SIZE,
        device=device,
        surrogate_name=surrogate_model_name,
    )

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
            args.attack, target_model_name, total_images, 
            acc_clean, acc_adv, score_asr, score_ssim, score_m
        )
        logging.info(report_table)

        # --- 5b. Archive Results ---
        if args.save_images:
            save_and_archive_results(
                attack_output_dir=attack_output_dir,
                label_file=LABEL_FILE,
                model_name=target_model_name,
                attack_name=args.attack
            )
    else:
        logging.error("No valid images were processed. Check dataset paths.")

if __name__ == "__main__":
    main()

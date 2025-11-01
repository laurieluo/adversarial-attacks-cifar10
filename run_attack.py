# run_attack.py
import torch
import os
import sys
import argparse
import shutil
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity

from src.models import ResNet18
from src.data_loader import get_custom_loader
from src.utils import get_device, NormalizedModel
from src.attacks import PGD, FGSM, BIM, CW, AutoAttack, Pixle, VNIFGSM, OnePixel, SparseFool, Jitter

# --- Configuration ---
MODEL_PATH = "saved_models/cifar10_resnet18.pth"
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
        choices=['pgd', 'fgsm', 'bim', 'cw', 'autoattack', 'pixle', 'vnifgsm', 'onepixel', 'sparsefool', 'jitter'],
        help="Type of attack to run (default: pgd)"
    )
    parser.add_argument(
        '--save_images',
        action='store_true',
        help="Save the generated adversarial images to disk"
    )
    return parser.parse_args()

# --- Main Execution ---
def main():
    args = parse_args()
    print(f"Running with attack: {args.attack.upper()}")

    # Define attack-specific output paths
    attack_output_dir = os.path.join(OUTPUT_DIR, args.attack.upper())
    images_save_dir = os.path.join(attack_output_dir, "images")
    
    # Only create directories and print if saving
    if args.save_images:
        os.makedirs(images_save_dir, exist_ok=True)
        print(f"Adversarial images will be saved to: {images_save_dir}")
    else:
        print("Note: Adversarial images will NOT be saved (use --save_images to save).")
    
    device = get_device()
    
    # Load the *base* pre-trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run 'python train.py' first to train and save the model.")
        print(f"Or, download the weights from https://huggingface.co/jaeunglee/resnet18-cifar10-unlearning/blob/main/resnet18_cifar10_full.pth")
        print(f"and rename it to '{MODEL_PATH}'.")
        sys.exit(1)
        
    print(f"Loading pre-trained model from: {MODEL_PATH}")
    base_model = ResNet18().to(device)
    
    try:
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    except RuntimeError as e:
        print("\n--- Warning: Weight loading failed (RuntimeError) ---")
        print(e)
        print("This usually means the architecture in 'src/models.py' does not match the '.pth' file.")
        print("Please ensure you have updated 'src/models.py' as per the previous response.")
        sys.exit(1)
        
    base_model.eval()

    # Wrap the model for normalization
    norm_model = NormalizedModel(base_model).to(device)
    norm_model.eval()

    # Load CIFAR10 500-image dataset
    print(f"Loading custom dataset from: {IMAGE_DIR}")
    custom_loader = get_custom_loader(
        image_dir=IMAGE_DIR, 
        label_file=LABEL_FILE, 
        batch_size=BATCH_SIZE
    )

    # --- Initialize the selected attack ---
    if args.attack == 'pgd':
        print("Initializing PGD attack...")
        atk = PGD(norm_model, eps=8/255, alpha=2/255, steps=10)
    elif args.attack == 'fgsm':
        print("Initializing FGSM attack...")
        atk = FGSM(norm_model, eps=8/255)
    elif args.attack == 'bim':
        print("Initializing BIM attack...")
        atk = BIM(norm_model, eps=8/255, alpha=2/255, steps=10)
    elif args.attack == 'cw':
        print("Initializing CW attack... (This may be very slow!)")
        atk = CW(norm_model, c=1, kappa=0, steps=1000, lr=0.01)
    elif args.attack == 'autoattack':
        print("Initializing AutoAttack... (This will be slow!)")
        atk = AutoAttack(norm_model, norm='Linf', eps=8/255)
    elif args.attack == 'pixle':
        print("Initializing Pixle Attack... (This may be slow due to optimization)")
        atk = Pixle(norm_model, pixel_mapping='similarity_random', restarts=20, max_iterations=10)
    elif args.attack == 'vnifgsm':
        print("Initializing VNIFGSM Attack...")
        atk = VNIFGSM(norm_model, eps=8/255, alpha=2/255, steps=10, decay=1.0)
    elif args.attack == 'onepixel':
        print("Initializing OnePixel Attack... (This will be EXTREMELY slow!)")
        atk = OnePixel(norm_model, pixels=1, steps=10, popsize=10, inf_batch=BATCH_SIZE)
    elif args.attack == 'sparsefool':
        print("Initializing SparseFool Attack... (This may be slow)")
        atk = SparseFool(norm_model, steps=10, lam=3, overshoot=0.02)
    elif args.attack == 'jitter': 
        print("Initializing Jitter Attack...")
        atk = Jitter(norm_model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)
    else:
        print(f"Error: Unknown attack '{args.attack}'")
        sys.exit(1)

    # Run evaluation and attack
    total_correct_clean = 0
    total_correct_adv = 0
    total_images = 0
    total_ssim_sum = 0.0

    progress_bar = tqdm(custom_loader, desc=f"Attacking ({args.attack.upper()})", leave=True)
    
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

        print("\n" + "="*30)
        print(f"Attack Evaluation Complete ({args.attack.upper()})")
        print(f"Total images tested: {total_images}")
        
        print("\n--- Standard Metrics ---")
        print(f"Model accuracy on CLEAN images: {acc_clean:.2f}%")
        print(f"Model accuracy on ADVERSARIAL images: {acc_adv:.2f}%")
        
        print("\n--- Custom Scores ---")
        print(f"Score_ASR (Attack Success Rate): {score_asr:.4f} ({(score_asr*100):.2f}%)")
        print(f"Score_SSIM (Average Structural Similarity): {score_ssim:.4f}")
        print(f"Score_M (Composite Score): {score_m:.4f}")

        if args.save_images:
            try:
                label_save_path = os.path.join(attack_output_dir, "label.txt")
                shutil.copy(LABEL_FILE, label_save_path)
                print(f"\nCopied label file to: {label_save_path}")
            except Exception as e:
                print(f"\nWarning: Could not copy label file. {e}")

    else:
        print("Error: No valid images were processed. Check dataset paths.")

if __name__ == "__main__":
    main()
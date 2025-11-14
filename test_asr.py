# test_asr.py
"""
Test Attack Success Rate (ASR) of adversarial images on one or more models.

Usage:
    python test_asr.py --adv-images <path_to_adversarial_images> --label-file <path_to_label_file> --models resnet18 vgg16
    python test_asr.py --adv-images adversarial_images/WRN2810/RFA_INF/images --label-file adversarial_images/WRN2810/RFA_INF/label.txt --models wrn2810 wrn9416
"""
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

import argparse
import torch

from src.utils import (
    get_device,
    NormalizedModel,
    load_model,
    load_adversarial_images,
    calculate_asr
)

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Attack Success Rate (ASR) of adversarial images on models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on a single model
  python test_asr.py --adv-images adversarial_images/WRN2810/RFA_INF/images --label-file adversarial_images/WRN2810/RFA_INF/label.txt --models resnet18
  
  # Test on multiple models
  python test_asr.py --adv-images adversarial_images/WRN2810/RFA_INF/images --label-file adversarial_images/WRN2810/RFA_INF/label.txt --models resnet18 vgg16 densenet121
  
  # Test with custom batch size
  python test_asr.py --adv-images adversarial_images/WRN2810/RFA_INF/images --label-file adversarial_images/WRN2810/RFA_INF/label.txt --models wrn2810 --batch-size 64
        """
    )
    parser.add_argument(
        '--adv-images', type=str, required=True,
        help="Path to directory containing adversarial images"
    )
    parser.add_argument(
        '--label-file', type=str, required=True,
        help="Path to label file (same format as clean dataset)"
    )
    parser.add_argument(
        '--models', type=str, nargs='+', required=True,
        choices=['resnet18', 'vgg16', 'densenet121', 'wrn2810', 'wrn9416'],
        help="One or more model names to test ASR on"
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help="Batch size for evaluation (default: 32)"
    )
    return parser.parse_args()


def print_asr_results_table(model_name, results):
    """
    Prints ASR results in a formatted table.
    """
    asr = results['asr']
    accuracy = results['accuracy']
    total_images = results['total_images']
    successful_attacks = results['successful_attacks']
    
    max_width = 59
    col1_width = 32
    col2_width = 20
    inner_width = max_width - 2
    
    title = f" ASR TEST: {model_name.upper()} "
    title_padding = (inner_width - len(title)) // 2
    title_padding_rem = inner_width - len(title) - title_padding
    
    if title_padding < 0:
        max_width = len(title) + 4
        inner_width = max_width - 2
        col1_width = 32
        col2_width = max_width - col1_width - 7
        title_padding = 1
        title_padding_rem = 1
    
    table = "\n"
    table += f"+{'=' * (max_width - 2)}+\n"
    table += f"|{' ' * title_padding}{title}{' ' * title_padding_rem}|\n"
    table += f"+{'=' * (max_width - 2)}+\n"
    table += f"| {'Metric':<{col1_width}} | {'Value':<{col2_width}} |\n"
    table += f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|\n"
    table += f"| {'Total Images Tested':<{col1_width}} | {total_images:<{col2_width}} |\n"
    table += f"| {'Successful Attacks':<{col1_width}} | {successful_attacks:<{col2_width}} |\n"
    acc_str = f"{accuracy:.2f}%"
    table += f"| {'Adversarial Accuracy':<{col1_width}} | {acc_str:<{col2_width}} |\n"
    asr_str = f"{asr:.4f} ({(asr*100):.2f}%)"
    table += f"| {'Attack Success Rate (ASR)':<{col1_width}} | {asr_str:<{col2_width}} |\n"
    table += f"+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+\n"
    
    return table


# --- Main Execution ---
def main():
    setup_logging()
    args = parse_args()
    
    # Validate paths
    if not os.path.isdir(args.adv_images):
        logging.error(f"Adversarial images directory not found: {args.adv_images}")
        sys.exit(1)
    
    if not os.path.exists(args.label_file):
        logging.error(f"Label file not found: {args.label_file}")
        sys.exit(1)
    
    logging.info(f"Testing ASR on {len(args.models)} model(s): {', '.join(args.models)}")
    logging.info(f"Adversarial images directory: {args.adv_images}")
    logging.info(f"Label file: {args.label_file}")
    logging.info(f"Batch size: {args.batch_size}")
    
    device = get_device()
    
    # Load adversarial images
    logging.info("Loading adversarial images...")
    adv_loader = load_adversarial_images(
        adv_image_dir=args.adv_images,
        label_file=args.label_file,
        batch_size=args.batch_size
    )
    
    # Test on each model
    all_results = {}
    
    for model_name in args.models:
        test_msg = f"Testing on model: {model_name.upper()}"
        logging.info(test_msg)
        
        # Load model
        try:
            base_model = load_model(model_name, device)
            norm_model = NormalizedModel(base_model).to(device)
            norm_model.eval()
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            continue
        
        # Calculate ASR
        logging.info("Calculating ASR...")
        results = calculate_asr(norm_model, adv_loader, device)
        all_results[model_name] = results
        
        # Print results
        result_table = print_asr_results_table(model_name, results)
        logging.info(result_table)
    
    # Print summary if multiple models
    if len(args.models) > 1:
        summary_msg = "SUMMARY - ASR Comparison Across Models"
        logging.info(f"{'='*60}")
        logging.info(summary_msg)
        logging.info(f"{'='*60}")
        
        # Calculate column widths for proper alignment
        model_col_width = max(20, max(len(m.upper()) for m in args.models) + 2)
        asr_col_width = 25
        acc_col_width = 15
        total_col_width = 15
        
        summary_table = "\n"
        summary_table += f"{'Model':<{model_col_width}} | {'ASR':<{asr_col_width}} | {'Accuracy':<{acc_col_width}} | {'Total Images':<{total_col_width}}\n"
        summary_table += f"{'-'*model_col_width}-+-{'-'*asr_col_width}-+-{'-'*acc_col_width}-+-{'-'*total_col_width}\n"
        
        total_asr_sum = 0.0
        valid_model_count = 0
        
        for model_name in args.models:
            if model_name in all_results:
                results = all_results[model_name]
                asr_str = f"{results['asr']:.4f} ({results['asr']*100:.2f}%)"
                acc_str = f"{results['accuracy']:.2f}%"
                total_str = f"{results['total_images']}"
                summary_table += f"{model_name.upper():<{model_col_width}} | {asr_str:<{asr_col_width}} | {acc_str:<{acc_col_width}} | {total_str:<{total_col_width}}\n"
                total_asr_sum += results['asr']
                valid_model_count += 1
        
        # Add average ASR row
        if valid_model_count > 0:
            avg_asr = total_asr_sum / valid_model_count
            avg_asr_str = f"{avg_asr:.4f} ({avg_asr*100:.2f}%)"
            summary_table += f"{'-'*model_col_width}-+-{'-'*asr_col_width}-+-{'-'*acc_col_width}-+-{'-'*total_col_width}\n"
            summary_table += f"{'Average ASR':<{model_col_width}} | {avg_asr_str:<{asr_col_width}} | {'':<{acc_col_width}} | {'':<{total_col_width}}\n"
        
        logging.info(summary_table)
    
    logging.info("\nASR testing completed!")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
The Ultimate Attempt: Admix Ensemble
Components:
1. Ensemble: Bartoldson (WRN)
2. Loss: DLR (AutoAttack) - No CrossEntropy!
3. Input: Admix + DI (SOTA Transfer)
4. Grad: Normalized Ensemble (Bartoldson dominates)
"""

from __future__ import annotations
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor
import timm # Ensure timm is installed
try:
    from robustbench.utils import load_model
except ImportError:
    pass # Will handle manually if needed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# --- Constants ---
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# --- SSIM Utils ---
def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.view(1, 1, -1)
    window_2d = window_1d.transpose(2, 1) @ window_1d
    return window_2d

def ssim_batch(x, y, window_size=11, sigma=1.5):
    device = x.device
    ws = window_size
    window = _gaussian_window(ws, sigma, device=device).view(1, 1, ws, ws).repeat(3, 1, 1, 1)
    mu_x = F.conv2d(x, window, padding=ws//2, groups=3)
    mu_y = F.conv2d(y, window, padding=ws//2, groups=3)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x*x, window, padding=ws//2, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y*y, window, padding=ws//2, groups=3) - mu_y2
    sigma_xy = F.conv2d(x*y, window, padding=ws//2, groups=3) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2))/((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map.mean(dim=(1,2,3))

# --- Model Loading Utils ---
# (Simplified for brevity, assuming you have the files locally as before)
class NormalizedModel(nn.Module):
    def __init__(self, model, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

def load_models(models_dir, device):
    models = []
    
    # 1. Bartoldson (WRN)
    print("Loading Bartoldson...")
    try:
        # Try local load first if you have the class defs, otherwise robustbench
        from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet
        m1 = DMWideResNet(num_classes=10, depth=94, width=16, activation_fn=nn.SiLU)
        sd = torch.load(os.path.join(models_dir, "Bartoldson2024Adversarial_WRN-94-16.pt"), map_location='cpu')
        # Fix keys
        new_sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
        m1.load_state_dict(new_sd, strict=False)
        m1 = NormalizedModel(m1, mean=CIFAR10_MEAN, std=CIFAR10_STD).eval().to(device)
        models.append(m1)
    except Exception as e:
        print(f"Error loading Bartoldson: {e}")
        exit(1)

    return models

# --- Core Logic: Admix + DI ---
def admix_transform(x, num_copies=3, portion=0.2):
    """
    Admix: Mix input x with a randomly permuted version of itself.
    This smooths the decision boundary for robust models.
    """
    x_admix = x.repeat(num_copies, 1, 1, 1) # [B*3, C, H, W]
    
    # Create permuted images
    indices = torch.randperm(x_admix.size(0), device=x.device)
    x_shuffled = x_admix[indices]
    
    # Mix: x' = x + lambda * x_shuffled
    # scaling down by (1+portion) to keep range roughly correct? 
    # Usually Admix uses simple addition then clip, but let's do convex combination for stability
    # x_out = x + portion * x_shuffled.
    return (x_admix + portion * x_shuffled).clamp(0, 1)

def input_diversity(x, prob=0.7):
    if torch.rand(1).item() > prob:
        return x
    rnd = int(torch.randint(32, 41, (1,)).item())  # 32..40
    if rnd != 32:
        x = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    pad = 40 - rnd
    if pad > 0:
        top = int(torch.randint(0, pad + 1, (1,)).item())
        left = int(torch.randint(0, pad + 1, (1,)).item())
        x = F.pad(x, (left, pad - left, top, pad - top))
    crop_top = int(torch.randint(0, 9, (1,)).item())  # 0..8
    crop_left = int(torch.randint(0, 9, (1,)).item())
    return x[:, :, crop_top:crop_top+32, crop_left:crop_left+32]

# --- DLR Loss ---
def dlr_loss(logits, y):
    z_sorted, z_indices = logits.sort(dim=1, descending=True)
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    z_p1 = z_sorted[:, 0]
    z_pi = torch.where(z_indices[:, 0] == y, z_sorted[:, 1], z_sorted[:, 0])
    z_p3 = z_sorted[:, 2]
    numerator = z_y - z_pi
    denominator = z_p1 - z_p3 + 1e-12
    # We want to minimize (z_y - z_pi), so maximize -(z_y - z_pi)
    # Return positive scalar for minimization? No, we do gradient ascent usually.
    # Let's return the Loss we want to MAXIMIZE.
    # We want to MAXIMIZE (z_pi - z_y).
    return (z_pi - z_y) / denominator

# --- Attack Loop ---
def run_admix_ensemble(
    x,
    y,
    models,
    eps,
    alpha,
    steps,
    mu=1.0,
    random_start=True,
    admix_copies=3,
    admix_portion=0.2,
    di_prob=0.7,
):
    x_adv = x.clone().detach()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    g_mom = torch.zeros_like(x_adv)
    
    # Ensemble weights: give Bartoldson higher weight if present
    if len(models) == 1:
        weights = [1.0]
    else:
        weights = [2.0] + [1.0] * (len(models) - 1)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        # 1. Admix Transform (computes gradient on 3 mixed copies)
        x_admix = admix_transform(x_adv, num_copies=admix_copies, portion=admix_portion)
        y_rep = y.repeat(admix_copies)
        
        # 2. DI (on top of Admix)
        x_in = input_diversity(x_admix, prob=di_prob)
        
        ensemble_grad = torch.zeros_like(x_adv)
        
        # 3. Model Forward & Grad
        for i, m in enumerate(models):
            logits = m(x_in)
            
            # Use DLR Loss (Sum over batch)
            loss = dlr_loss(logits, y_rep).sum()
            
            # Grad
            g = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
            
            # Normalize
            g_norm = g.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            g = g / g_norm
            
            # Accumulate
            ensemble_grad += g * weights[i]

        # 4. Normalize
        final_norm = ensemble_grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        ensemble_grad /= final_norm
        
        # 5. Momentum
        g_mom = mu * g_mom + ensemble_grad
        
        # 6. Update
        x_adv = x_adv.detach() + alpha * g_mom.sign()
        delta = (x_adv - x).clamp(-eps, eps)
        x_adv = (x + delta).clamp(0, 1).detach()
        
    return x_adv

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/cifar10_clean_500")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default="adversarial_images/Admix_Ensemble")
    parser.add_argument("--batch-size", type=int, default=20)
    # Params
    parser.add_argument("--eps", type=float, default=16/255) 
    parser.add_argument("--steps", type=int, default=80) # Base steps (may be overridden per config)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running Admix Ensemble (Bart). Eps={args.eps:.4f}")

    models = load_models(args.models_dir, device)
    
    # Dataset
    class SimpleDataset(Dataset):
        def __init__(self, root):
            self.img_dir = os.path.join(root, "images")
            self.items = []
            with open(os.path.join(root, "label.txt")) as f:
                for l in f:
                    if l.strip(): self.items.append(l.strip().split())
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            p, l = self.items[i]
            img = Image.open(os.path.join(self.img_dir, p)).convert('RGB')
            return to_tensor(img), torch.tensor(int(l)), p
            
    ds = SimpleDataset(args.data_root)
    
    # Configurations to run sequentially
    configs = [
        {
            "name": "A_steps80_alpha12_r5_admix3_0.2",
            "steps": 80,
            "alpha_scale": 12.0,
            "restarts": 5,
            "admix_copies": 3,
            "admix_portion": 0.2,
        },
    ]
    
    for cfg in configs:
        print(f"\n[CONFIG] {cfg['name']}")
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        if tqdm: dl = tqdm(dl)
        
        correct_clean = 0
        correct_adv = 0
        ssim_sum = 0
        total = 0
        
        out_dir_cfg = os.path.join(args.out_dir, cfg["name"])
        
        for x, y, fnames in dl:
            x, y = x.to(device), y.to(device)
            
            # Attack with multiple restarts, keep the one with lowest true-class logit
            best_x_adv = None
            best_margin = None
            for _ in range(cfg["restarts"]):
                x_adv_candidate = run_admix_ensemble(
                    x,
                    y,
                    models,
                    args.eps,
                    alpha=args.eps/cfg["steps"]*cfg["alpha_scale"],
                    steps=cfg["steps"],
                    random_start=True,
                    admix_copies=cfg["admix_copies"],
                    admix_portion=cfg["admix_portion"],
                    di_prob=0.7,
                )
                with torch.no_grad():
                    logits_candidate = models[0](x_adv_candidate)
                    margin = logits_candidate.gather(1, y.view(-1, 1)).squeeze(1)
                if best_x_adv is None:
                    best_x_adv = x_adv_candidate
                    best_margin = margin
                else:
                    better = margin < best_margin
                    best_x_adv = torch.where(better.view(-1, 1, 1, 1), x_adv_candidate, best_x_adv)
                    best_margin = torch.where(better, margin, best_margin)
            x_adv = best_x_adv
            
            # Eval
            with torch.no_grad():
                correct_clean += (models[0](x).argmax(1) == y).sum().item()
                correct_adv += (models[0](x_adv).argmax(1) == y).sum().item()
                ssim_sum += ssim_batch(x, x_adv).sum().item()
            
            total += x.size(0)
            
            # Save
            os.makedirs(out_dir_cfg, exist_ok=True)
            x_byte = (x_adv.clamp(0,1)*255).byte().cpu()
            for i, fn in enumerate(fnames):
                Image.fromarray(x_byte[i].permute(1,2,0).numpy()).save(os.path.join(out_dir_cfg, fn))
                
        print(f"[RESULT] {cfg['name']}")
        print(f"Clean Acc: {correct_clean/total:.4f}")
        print(f"Adv Acc (Local Bart): {correct_adv/total:.4f}")
        print(f"ASR (Local Bart): {1 - correct_adv/total:.4f}")
        print(f"SSIM: {ssim_sum/total:.4f}")

if __name__ == "__main__":
    main()
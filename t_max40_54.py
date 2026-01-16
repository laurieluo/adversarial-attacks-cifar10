#!/usr/bin/env python3
"""
The Ultimate Solution: Auto-Greedy Attack + Auto-Zip
Features:
1. Safety Net (Max Eps) -> Guarantees ASR.
2. Greedy Search (Small Eps) -> Maximizes SSIM.
3. Auto Zip -> Packages everything into 'images/' folder format for submission.
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
import zipfile # 引入zip库

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# --- Configuration ---
# 细粒度阶梯：从小到大
# 策略：只要小扰动能成功，就不用大扰动
EPS_LEVELS_255 = [16, 20, 24, 28, 32, 36, 40] 
MAX_EPS = 64.0 / 255.0
STEPS = 100 
DI_PROB = 0.9 

# --- Constants ---
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# --- Model Wrapper ---
class NormalizedModel(nn.Module):
    def __init__(self, model, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# --- Transformations ---
def admix_transform(x, num_copies=3, portion=0.2):
    x_admix = x.repeat(num_copies, 1, 1, 1)
    indices = torch.randperm(x_admix.size(0), device=x.device)
    x_shuffled = x_admix[indices]
    return (x_admix + portion * x_shuffled).clamp(0, 1)

def input_diversity(x, prob=0.5):
    if torch.rand(1).item() > prob: return x
    rnd = int(torch.randint(29, 33, (1,)).item())
    if rnd == 32: return x
    x = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    pad = 32 - rnd
    top = int(torch.randint(0, pad + 1, (1,)).item())
    left = int(torch.randint(0, pad + 1, (1,)).item())
    x = F.pad(x, (left, pad - left, top, pad - top))
    return x

# --- DLR Loss ---
def dlr_loss(logits, y):
    z_sorted, z_indices = logits.sort(dim=1, descending=True)
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    z_p1 = z_sorted[:, 0]
    z_pi = torch.where(z_indices[:, 0] == y, z_sorted[:, 1], z_sorted[:, 0])
    z_p3 = z_sorted[:, 2]
    return (z_pi - z_y) / (z_p1 - z_p3 + 1e-12)

# --- Attack Core ---
def run_attack_batch(x, y, model, eps, steps):
    alpha = eps / steps * 2.5 
    x_adv = x.clone().detach()
    x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0, 1)
    g_mom = torch.zeros_like(x_adv)
    
    for t in range(steps):
        x_adv.requires_grad_(True)
        
        # Admix + DI
        x_admix = admix_transform(x_adv, num_copies=3, portion=0.2)
        x_in = input_diversity(x_admix, prob=DI_PROB)
        y_rep = y.repeat(3)
        
        logits = model(x_in)
        loss = dlr_loss(logits, y_rep).sum()
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # Normalize & Momentum
        g_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        grad = grad / g_norm
        g_mom = 0.9 * g_mom + grad 
        
        # Update
        x_adv = x_adv.detach() + alpha * g_mom.sign()
        delta = (x_adv - x).clamp(-eps, eps)
        x_adv = (x + delta).clamp(0, 1).detach()
        
    return x_adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./dataset/cifar10_clean_500")
    parser.add_argument("--models-dir", default="./saved_models")
    parser.add_argument("--out-dir", default="submission_images_max40")
    parser.add_argument("--zip-name", default="submission_max40.zip")   
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running Fine-Grained Greedy Strategy on {device}...")

    # 1. Load Model
    from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet
    m_raw = DMWideResNet(num_classes=10, depth=94, width=16, activation_fn=nn.SiLU)
    model_path = os.path.join(args.models_dir, "Bartoldson2024Adversarial_WRN-94-16.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    sd = torch.load(model_path, map_location='cpu')
    if 'state_dict' in sd: sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m_raw.load_state_dict(sd, strict=False)
    model = NormalizedModel(m_raw).eval().to(device)

    # 2. Dataset
    class SimpleDataset(Dataset):
        def __init__(self, root):
            self.img_dir = os.path.join(root, "images")
            self.items = []
            if os.path.exists(os.path.join(root, "label.txt")):
                with open(os.path.join(root, "label.txt")) as f:
                    for l in f:
                        if l.strip(): self.items.append(l.strip().split())
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            p, l = self.items[i]
            img = Image.open(os.path.join(self.img_dir, p)).convert('RGB')
            return to_tensor(img), torch.tensor(int(l)), p

    ds = SimpleDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    if tqdm: dl = tqdm(dl)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    total = 0
    clean_correct = 0
    final_adv_correct = 0
    
    # 3. Strategy Loop
    eps_list = [e / 255.0 for e in EPS_LEVELS_255]
    
    for x, y, fnames in dl:
        x, y = x.to(device), y.to(device)
        B = x.size(0)
        total += B
        
        # A. Clean Acc
        with torch.no_grad():
            clean_acc = (model(x).argmax(1) == y)
            clean_correct += clean_acc.sum().item()

        # B. Safety Net (Max Eps)
        x_safety = run_attack_batch(x, y, model, eps=MAX_EPS, steps=STEPS)
        final_x = x_safety.clone()
        solved_mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        # C. Greedy Search
        for eps in eps_list:
            if solved_mask.all(): break
            
            x_candidate = run_attack_batch(x, y, model, eps=eps, steps=STEPS)
            
            with torch.no_grad():
                pred = model(x_candidate).argmax(1)
                is_adv = (pred != y) 
                
                # Update if attack successful AND not already solved by smaller eps
                update_mask = is_adv & (~solved_mask)
                
                if update_mask.sum() > 0:
                    mask_broad = update_mask.view(B, 1, 1, 1)
                    final_x = torch.where(mask_broad, x_candidate, final_x)
                    solved_mask = solved_mask | update_mask
                    
        # D. Save Images
        with torch.no_grad():
            final_pred = model(final_x).argmax(1)
            final_adv_correct += (final_pred == y).sum().item()
            
        x_byte = (final_x.clamp(0,1)*255).byte().cpu()
        for i, fn in enumerate(fnames):
            file_name = os.path.basename(fn)
            Image.fromarray(x_byte[i].permute(1,2,0).numpy()).save(os.path.join(args.out_dir, file_name))
            
    # 4. Stats
    asr = 1 - final_adv_correct/total
    print(f"\n[ATTACK FINISHED]")
    print(f"Total: {total}")
    print(f"Clean Acc: {clean_correct/total:.4f}")
    print(f"Final ASR: {asr:.4f}")
    print(f"Strategy: Safety Net ({MAX_EPS:.4f}) -> Greedy Refinement ({eps_list[0]:.4f} to {eps_list[-1]:.4f})")
    
    # 5. Auto Zip (Format: images/xxx.png)
    print(f"\n[ZIP] Compressing to {args.zip_name}...")
    with zipfile.ZipFile(args.zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.out_dir):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('images', file)
                    zf.write(file_path, arcname)
                    
    print(f"[DONE] File ready for submission: {args.zip_name}")

if __name__ == "__main__":
    main()

# adversarial-attacks-cifar10

## 📂 Project Structure

```bash
.
├── adversarial_images/      # (Generated) Saves adversarial images and ZIPs
│   └── [MODEL_NAME]/
│       └── [ATTACK_NAME]/
│           ├── images/      # 500 adversarial images
│           ├── label.txt    # Label file
│           └── [MODEL]_[ATTACK]_[timestamp].zip
│
├── dataset/
│   └── cifar10_clean_500/   # (Manual) 500 images for attacking
│       ├── images/
│       └── label.txt
│   # (train.py auto-downloads full CIFAR-10 here)
│
├── saved_models/            # (Generated) Saved trained model weights
│   ├── cifar10_resnet18.pth
│   ├── cifar10_vgg16.pth
│   └── cifar10_densenet121.pth
│
├── src/                     # Source code directory
│   ├── attacks/             # Attack wrappers (pgd.py, fgsm.py, ...)
│   ├── __init__.py
│   ├── data_loader.py       # Data loaders
│   ├── logger.py            # Colored logger configuration
│   ├── models.py            # Model definitions (ResNet18, VGG16, ...)
│   └── utils.py             # Utility functions 
│
├── README.md                # Project description
├── requirements.txt         # Python dependencies
├── run_attack.py            # Main script to run adversarial attacks
├── test_asr.py              # Script to test ASR on adversarial images
└── train.py                 # Main script to train new models
```

## 🛠 Usage

### 1. Environment Setup

First, set up the project environment using Conda and install the required packages.

```bash
# Create a new conda environment
$ conda create -n adv pip python=3.10

# Activate the environment
$ conda activate adv

# Install all required packages
$ pip install -r requirements.txt
```

#### TransferAttack

```bash
# Run at the project root when first start:
$ git clone https://github.com/Trustworthy-AI-Group/TransferAttack.git
```

### 2. Prepare Data

You only need the 500-image attack dataset for attacking.
During training, the full CIFAR-10 training set will be downloaded automatically.

```bash
# Create the main dataset directory
$ mkdir dataset

# 
# Manually download and place your 'cifar10_clean_500' directory
# into the 'dataset' folder.
#
# The final structure should be:
# ./dataset/cifar10_clean_500/
# ├── images/
# │   ├── 0.png
# │   ├── 1.png
# │   └── ...
# └── label.txt
#
```

### 3. (Optional) Train/Download Models

You can train a new model. The `train.py` script will save the model to the saved_models/ directory.
**ONLY** when you want to add a new model, you need to define the model in `src/models.py` and modify the code
in `train.py`.

**Note**: We do not provide pre-trained models. You need to train or download them yourself.

#### Download robust models:

Put them in the `saved_models/` folder:
- [Bartoldson2024Adversarial_WRN-94-16.pt (1.4G)](https://drive.usercontent.google.com/download?id=1g6o9H1b6vjoBi1USdCBt64C8B8LPiioX&authuser=0)
- [Cui2023Decoupled_wrn-28-10.pt (139M)](https://drive.usercontent.google.com/download?id=1-AaTrYt23WJFR22hXgBd-i6kjpsz6Hf2&authuser=0)
- [Wang2023Better_wrn-70-16.pt (1,018M)](https://drive.google.com/uc?id=1-RF7ZSS-PAh6bfQcuqx4lh9bc9BUGnap)

#### Example usage:

```bash
# Train the default ResNet-18 model (takes ~10-15 min on a good GPU)
$ python train.py --model resnet18

# You can also train other architectures:
# $ python train.py --model vgg16
# $ python train.py --model densenet121
```

### 4. Run an Attack

Once you have a saved model (e.g., saved_models/cifar10_resnet18.pth), you can run attacks against it using run_attack.py.

```bash
# 1. Run a PGD attack on ResNet-18 (no images saved)
$ python run_attack.py --model resnet18 --attack pgd

# 2. Run a FGSM attack on VGG-16 and save the adversarial images
#    This will create a ZIP file in:
#    adversarial_images/VGG16/FGSM/VGG16_FGSM_[timestamp].zip
$ python run_attack.py --model vgg16 --attack fgsm --save-images

# 3. Run a integrated attack algorithm like adaea,if save images,
#    it will save to adversarial_images/RESNET18
$ python run_attack.py --attack adaea
$ python run_attack.py --attack adaea --save-images

# 4. Run RFA_inf attack on robust model
$ python run_attack.py --attack rfa_inf --model wrn9416 --target-model wrn9416 --save-images
```

### 5. Test ASR on Adversarial Images

After generating adversarial images, you can test their Attack Success Rate (ASR) on one or more models using `test_asr.py`.

```bash
# 1. Test ASR on a single model
$ python test_asr.py --adv-images adversarial_images/WRN2810/RFA_INF/images --label-file adversarial_images/WRN2810/RFA_INF/label.txt --models resnet18

# 2. Test ASR on multiple models (compare transferability)
$ python test_asr.py --adv-images adversarial_images/WRN2810/RFA_INF/images --label-file adversarial_images/WRN2810/RFA_INF/label.txt --models resnet18 vgg16 densenet121 wrn2810 wrn9416

# 3. Test with custom batch size
$ python test_asr.py --adv-images adversarial_images/WRN2810/RFA_INF/images --label-file adversarial_images/WRN2810/RFA_INF/label.txt --models wrn2810 --batch-size 64
```

**Parameters:**
- `--adv-images`: Path to directory containing adversarial images (required)
- `--label-file`: Path to label file (same format as clean dataset, required)
- `--models`: One or more model names to test ASR on (required, choices: resnet18, vgg16, densenet121, wrn2810, wrn9416, wrn7016)
- `--batch-size`: Batch size for evaluation (optional, default: 32)

**Output:**
- Individual ASR results for each model tested
- Summary table comparing ASR across all models (when multiple models are tested)
- Average ASR across all models

## ✅ Current Results

- **Attack Methods**: Based on [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).

### 1 ResNet18 Results

- **Accuracy on Clean Dataset**: 97%

#### 1.1 TorchAttack Methods

| Attack     | Score_ASR | Score_SSIM | Score_M | Platform Score |
|------------|-----------|------------|---------|----------------|
| PGD        | 0.9980    | 0.9614     | 95.9453 | _11.3433_      |
| FGSM       | 0.4960    | 0.9272     | 45.9895 | _not test_     |
| BIM        | 0.9980    | 0.9642     | 96.2294 | _9.1243_       |
| CW         | 1.0000    | 0.9974     | 99.7352 | _9.8150_       |
| AutoAttack | 1.0000    | 0.9620     | 96.1970 | _10.0519_      |
| **Pixle**  | 1.0000    | 0.7626     | 76.2641 | **_20.7329_**  |
| VNIFGSM    | 0.9980    | 0.9375     | 93.5585 | _10.3576_      |
| OnePixel   | 0.2900    | 0.9610     | 27.8702 | _not test_     |
| SparseFool | 0.7940    | 0.2443     | 19.3946 | _not test_     |
| Jitter     | 0.8360    | 0.9636     | 80.5606 | _10.3559_      |
| AIFGTM     | 0.9920    | 0.9652     | 95.7507 | _7.2961_       |

#### 1.2 TransferAttack OPS

| Model     | $\epsilon$ | $\alpha$ | $N_e$ | $N_p$ | ASR | SSIM | Score |
|-----------|------------|----------|-------|-------|-----|------|-------|
|ResNet18   | 0.16       |0.016     |20     |20     |1    |0.6050|25.7335|
|ResNet18   | 0.18       |0.018     |20     |20     |1    |0.5286|25.3164|
|ResNet18   | 0.16       |0.016     |30     |30     |1    |0.6023|25.4922|
|ResNet18   | 0.14       |0.014     |20     |20     |1    |0.6485|25.9854|
|ResNet18   | 0.12       |0.012     |20     |20     |1    |0.6957|24.6067|
|VGG16      | 0.14       |0.014     |20     |20     |1    |0.6485|26.7114|
|DenseNet121| 0.14       |0.014     |20     |20     |1    |-     |26.2286|

#### 1.3 Pixle Attack Parameters Optimizing

| Model    | dim. | pixel_map        | res. | max_iter | ASR | SSIM | M | Score |
| -------- | --------- | -------------------- | -------- | -------------- | --- | ---- | - | -------------- |
| ResNet18 | (2, 10)   | random               | 20       | 10             | 1.0000 | 0.7626 | 76.2641 | _20.7329_ |
| ResNet18 | 1         | random               | 100      | 20             | 1.0000 | 0.9325 | 93.2459 | _7.5789_ |
| ResNet18 | 1         | random               | 100      | 50             | 1.0000 | 0.9403 | 94.0294 | _8.1837_ |
| ResNet18 | 2         | random               | 100      | 50             | 1.0000 | 0.9121 | 91.2135 | _not test_ |
| ResNet18 | 3         | random               | 100      | 50             | 1.0000 | 0.8847 | 88.4733 | _9.8378_ |
| ResNet18 | 3         | simi.           | 100      | 50             | 0.9920 | 0.9046 | 89.7346 | _not test_ |
| ResNet18 | 3         | s_r    | 100      | 50             | 1.0000 | 0.8892 | 88.9165 | _not test_ |
| ResNet18 | (2, 10)   | random               | 100      | 50             | - | - | - | _16.4624_ |
| ResNet18 | (5, 15)   | random               | 20       | 10             | 0.9980 | 0.6849 | 68.3518 | _22.5834_ |
| ResNet18 | (10, 20)  | random               | 20       | 10             | 1.0000 | 0.6170 | 61.7004 | _24.7218_ |

### 2 Robust Model Results

- **Accuracy on Clean Dataset**: 98.2%

- **TorchAttackEval Methods**: [TransferAttackEval](https://github.com/ZhengyuZhao/TransferAttackEval/tree/main)
- **Robust Models**: [Robust Bench](https://robustbench.github.io/#leaderboard)

|      Model                        | Attack     |   ASR   |  SSIM  |    M    |    Score    |
|-----------------------------------|------------|---------|--------|---------|-------------|
|Bartoldson2024Adversarial_WRN-94-16| RFA-inf| 0.9180  | 0.7502 | 68.8643 |_**43.0488**_|
|Cui2023Decoupled_wrn-28-10         | RFA-inf|       - |      - |       - |_38.1566_    |
|Bartoldson2024Adversarial_WRN-94-16| GRA        | 0.2120  | 0.8844 | 18.7497 |_32.5697_|
|Bartoldson2024Adversarial_WRN-94-16| PGN        | 0.1980  | 0.8830 | 17.4841 |_32.9981_|
|Bartoldson2024Adversarial_WRN-94-16| MEF        | 0.2020  | 0.8723 | 17.6204 |_33.4605_|

#### 2.1 RFA∞ Attack Results on Multiple Models

| Model                | ASR                       | Accuracy        | Total Images   |
|----------------------|---------------------------|-----------------|----------------|
| RESNET18             | 0.9400 (94.00%)           | 6.00%           | 500            |
| VGG16                | 0.9120 (91.20%)           | 8.80%           | 500            |
| DENSENET121          | 0.9320 (93.20%)           | 6.80%           | 500            |
| WRN2810              | 0.8880 (88.80%)           | 11.20%          | 500            |
| WRN9416              | 0.8100 (81.00%)           | 19.00%          | 500            |
| WRN7016              | 0.9180 (91.80%)           | 8.20%           | 500            |
|----------------------|---------------------------|-----------------|----------------|
| Average ASR          | 0.9000 (90.00%)           |                 |                |

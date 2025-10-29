# adversarial-attacks-cifar10

## ✅ Current Results

- **Model**: ResNet18
- **Accuracy on Clean Dataset**: 97%
- **Attack Methods**: [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)

| Attack | Score_ASR | Score_SSIM | Score_M | Platform Score |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| PGD | 0.9980 | 0.9614 | 95.9453 | _11.3433_ |
| FGSM | 0.4960 | 0.9272 | 45.9895 | _Pass_ |
| BIM | 0.9980 | 0.9642 | 96.2294 | _9.1243_ |
| CW | 1.0000 | 0.9974 | 99.7352 | _9.8150_ |
| AutoAttack | 1.0000 | 0.9620 | 96.1970 | _10.0519_ |
| Pixle | 1.0000 | 0.7626 | 76.2641 | **_20.7329_** |
| VNIFGSM | 0.9980 | 0.9375 | 93.5585 | _Waiting..._ |

## 🛠️ Usage

1. `$ mkdir dataset`
1. Put cifar10_clean_500 dir into dataset
1. `$ conda create -n adv pip python=3.10`
1. `$ conda activate adv`
2. `$ pip install -r requirements.txt`
3. `$ python run_attack.py --attack pgd --save-images` or `$ python run_attack.py --attack pgd`
   without saving adversarial images.

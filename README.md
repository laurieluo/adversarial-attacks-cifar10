# adversarial-attacks-cifar10

## 🛠️ Usage

1. `$ mkdir dataset`
1. Put cifar10_clean_500 dir into dataset
1. `$ conda create -n adv pip python=3.10`
1. `$ conda activate adv`
2. `$ pip install -r requirements.txt`
3. `$ python run_attack.py --attack pgd --save-images` or `$ python run_attack.py --attack pgd`
   without saving adversarial images.

## ✅ Current Results

- **Attack Methods**: Based on [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).

### ResNet18 Results

- **Accuracy on Clean Dataset**: 97%

| Attack | Score_ASR | Score_SSIM | Score_M | Platform Score |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| PGD | 0.9980 | 0.9614 | 95.9453 | _11.3433_ |
| FGSM | 0.4960 | 0.9272 | 45.9895 | _Pass_ |
| BIM | 0.9980 | 0.9642 | 96.2294 | _9.1243_ |
| CW | 1.0000 | 0.9974 | 99.7352 | _9.8150_ |
| AutoAttack | 1.0000 | 0.9620 | 96.1970 | _10.0519_ |
| **Pixle** | 1.0000 | 0.7626 | 76.2641 | **_20.7329_** |
| VNIFGSM | 0.9980 | 0.9375 | 93.5585 | _10.3576_ |
| OnePixel | 0.2900 | 0.9610 | 27.8702 | _Pass_ |
| SparseFool | 0.7940 | 0.2443 | 19.3946 | _Pass_ |
| Jitter | 0.8360 | 0.9636 | 80.5606 | _10.3_ |

### Pixle Attack Parameters Optimizing

| Model    | dimension | pixel_mapping        | restarts | max_iterations | ASR | SSIM | M | Platform Score |
| -------- | --------- | -------------------- | -------- | -------------- | --- | ---- | - | -------------- |
| ResNet18 | (2, 10)   | random               | 20       | 10             | 1.0000 | 0.7626 | 76.2641 | 20.7329 |
| ResNet18 | 1         | random               | 100      | 20             | 1.0000 | 0.9325 | 93.2459 | 7.5789 |
| ResNet18 | 1         | random               | 100      | 50             | 1.0000 | 0.9403 | 94.0294 | 8.1837 |
| ResNet18 | 2         | random               | 100      | 50             | 1.0000 | 0.9121 | 91.2135 | Pass |
| ResNet18 | 3         | random               | 100      | 50             | 1.0000 | 0.8847 | 88.4733 | 9.8378 |
| ResNet18 | 3         | similarity           | 100      | 50             | 0.9920 | 0.9046 | 89.7346 | Pass |
| ResNet18 | 3         | similarity_random    | 100      | 50             | 1.0000 | 0.8892 | 88.9165 | Pass |

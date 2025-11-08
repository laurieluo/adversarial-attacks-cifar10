import torchattacks
import logging
from .base_attack import BaseAttack
import torch

import torchvision.transforms as T
from torchvision.transforms.functional import gaussian_blur


class PGD_CW(BaseAttack):
    """
    混合PGD与CW的对抗攻击算法：
    1. 第一步：用PGD快速生成接近决策边界的初始对抗样本
    2. 第二步：用CW在初始样本基础上进行L2优化，提升SSIM
    """
    def __init__(
            self,
            model,
            # PGD参数（沿用项目经典配置，保证速度与基础攻击效果）
            pgd_eps=8 / 255,
            pgd_alpha=2 / 255,
            pgd_steps=10,
            pgd_random_start=True,
            # CW参数（精简步数，利用PGD初始结果减少优化成本）
            cw_c=0.5,
            cw_kappa=0,
            cw_steps=200,  # 远少于纯CW的1000步，提升速度
            cw_lr=0.01,
            aug_degrees = 5,  # 随机旋转角度范围（±5°，避免语义篡改）
            aug_translate = 0.05,  # 随机平移比例（±5%）
            aug_scale = (0.95, 1.05),  # 随机缩放范围（0.95~1.05倍）
            blur_kernel_size = (3, 3),  # 高斯滤波核大小（3x3，轻度平滑）
            blur_sigma = (0.3, 0.5)  # 高斯滤波标准差（0.3~0.5，避免过度模糊）
    ):
        """
        :param model: 待攻击模型（需提前用NormalizedModel封装）
        :param pgd_eps: PGD的L∞最大扰动（默认8/255，符合CIFAR-10场景）
        :param pgd_alpha: PGD每步迭代步长（默认2/255）
        :param pgd_steps: PGD迭代步数（默认10步，快速收敛）
        :param pgd_random_start: PGD是否随机初始化扰动（默认True，提升攻击鲁棒性）
        :param cw_c: CW的拉格朗日系数（默认0.5，平衡扰动大小与攻击成功率）
        :param cw_kappa: CW的攻击置信度（默认0，仅需误分类即可）
        :param cw_steps: CW的优化步数（默认200步，比纯CW快5倍）
        :param cw_lr: CW的优化学习率（默认0.01，保证优化稳定性）
        """
        logging.info("Initializing PGDCW Hybrid Attack...")
        super().__init__(model)

        # 1. 初始化PGD攻击实例（负责快速接近决策边界）
        self.pgd_attack = torchattacks.PGD(
            model=model,
            eps=pgd_eps,
            alpha=pgd_alpha,
            steps=pgd_steps,
            random_start=pgd_random_start
        )
        logging.info(
            f"PGD Sub-Attack Config: "
            f"eps={pgd_eps:.4f}, alpha={pgd_alpha:.4f}, steps={pgd_steps}, random_start={pgd_random_start}"
        )

        # 2. 初始化CW攻击实例（负责L2微调，提升SSIM）
        self.cw_attack = torchattacks.CW(
            model=model,
            c=cw_c,
            kappa=cw_kappa,
            steps=cw_steps,
            lr=cw_lr
        )
        logging.info(
            f"CW Sub-Attack Config: "
            f"c={cw_c}, kappa={cw_kappa}, steps={cw_steps}, lr={cw_lr}"
        )
        logging.warning(
            "PGDCW Attack: CW steps reduced to 200 (from 1000) for speed; "
            "adjust cw_steps if SSIM needs further improvement."
        )

        # 3. 初始化随机数据增强（攻击前用，提升迁移性）
        # 注：仅用“旋转/平移/缩放”，避免裁剪导致语义丢失，符合竞赛合规性
        self.augmentor = T.RandomAffine(
            degrees=aug_degrees,
            translate=(aug_translate, aug_translate),
            scale=aug_scale,
            fill=0  # 平移/缩放后的空白用黑色填充（不影响主体语义）
        )
        logging.info(
            f"Data Augmentation Config: "
            f"degrees={aug_degrees}, translate={aug_translate}, scale={aug_scale}"
        )

        # 4. 初始化轻度高斯滤波（攻击后用，过滤噪声）
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size
        logging.info(
            f"Gaussian Blur Config: "
            f"kernel_size={blur_kernel_size}, sigma={blur_sigma}"
        )

    def attack(self, images, labels):
        """
        生成混合对抗样本：PGD初始化 → CW微调
        :param images: 清洁图像批次（Tensor，范围[0,1]，shape=(B,3,32,32)）
        :param labels: 图像真实标签（Tensor，shape=(B,)）
        :return: 最终混合对抗样本（Tensor，范围[0,1]）
        """
        # 确保输入图像在[0,1]范围（避免CW优化时数值溢出）
        images = torch.clamp(images, 0.0, 1.0)

        # -------------------------- 新增步骤：随机数据增强（提升迁移性）--------------------------
        # 对清洁图像施加小幅度随机变换，生成“增强后清洁图像”（不改变语义，符合合规性）
        images_aug = self.augmentor(images)
        logging.debug(f"Data Augmentation Completed: Image shape={images_aug.shape}")

        # 第一步：用PGD生成初始对抗样本（快速接近决策边界）
        x_adv_pgd = self.pgd_attack(images, labels)
        logging.debug(f"PGD Initialization Completed: Max Perturbation={torch.max(torch.abs(x_adv_pgd - images)):.4f}")

        # 第二步：用CW在PGD结果上微调（最小化L2扰动，提升SSIM）
        # 注：CW默认输入范围[0,1]，与x_adv_pgd一致，无需额外归一化
        x_adv_pgdcw = self.cw_attack(x_adv_pgd, labels)
        logging.debug(f"CW Fine-Tuning Completed: Max Perturbation={torch.max(torch.abs(x_adv_pgdcw - x_adv_pgd)):.4f}")

        # -------------------------- 新增步骤2：轻度高斯滤波（过滤肉眼可见噪声）--------------------------
        # 对CW微调后的样本进行滤波，平滑高频噪声（不影响攻击效果）
        # 注：gaussian_blur支持批次处理，无需循环
        x_adv_blur = gaussian_blur(
            img=x_adv_pgdcw,
            kernel_size=self.blur_kernel_size,
            sigma=self.blur_sigma
        )
        logging.debug(f"Gaussian Blur Completed: Image shape={x_adv_blur.shape}")

        # 最终裁剪（确保输出在[0,1]，避免极端值影响视觉质量）
        x_adv_final = torch.clamp(x_adv_blur, 0.0, 1.0)
        logging.debug(f"CW Fine-Tuning Completed: Max Perturbation={torch.max(torch.abs(x_adv_pgdcw - x_adv_pgd)):.4f}")

        return x_adv_pgdcw

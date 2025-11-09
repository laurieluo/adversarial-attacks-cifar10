# src/attacks/pixle_vnifgsm.py

import logging
import torch
from .base_attack import BaseAttack

try:
    from transferattack.gradient.vnifgsm import VNIFGSM as TransferAttackVNIFGSM
    import torchattacks
except ImportError as e:
    logging.error(
        "FATAL: Failed to import required packages for Pixle_VNIFGSM_Hybrid."
    )
    logging.error(f"Original error: {e}")
    raise e


class Pixle_VNIFGSM(BaseAttack):
    """
    混合攻击：先进行Pixle攻击（高比例），再进行VNIFGSM微调优化。

    基于比赛结果优化：
    - Pixle: 平台得分最高 (24.7218)
    - 策略：先用Pixle获得高得分，再用VNIFGSM提升SSIM
    """

    def __init__(self, model, eps, alpha, steps,
                 pixle_x_dimensions=(10, 20), pixle_y_dimensions=(10, 20),  # 使用得分最高的维度
                 pixel_mapping='random', pixle_restarts=20, pixle_max_iterations=10,  # 使用原始参数
                 vni_decay=1.0, vni_n=20, vni_beta=1.5,
                 pixle_ratio=0.8, vni_ratio=0.2):  # 提高Pixle比例到80%
        """
        初始化Pixle-VNIFGSM混合攻击（Pixle优先）。

        :param model: 要攻击的模型
        :param eps: L∞扰动上限
        :param alpha: VNIFGSM步长
        :param steps: 总迭代次数
        :param pixle_*: Pixle相关参数（使用比赛最优参数）
        :param vni_*: VNIFGSM相关参数
        :param pixle_ratio: Pixle阶段占比（提高至80%）
        :param vni_ratio: VNIFGSM阶段占比（降低至20%）
        """
        logging.info("Initializing Pixle-Prioritized Hybrid Attack...")
        super().__init__(model)

        # 基础参数
        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        # Pixle参数 - 使用比赛平台得分最高的配置
        self.pixle_x_dimensions = pixle_x_dimensions
        self.pixle_y_dimensions = pixle_y_dimensions
        self.pixle_pixel_mapping = pixel_mapping
        self.pixle_restarts = pixle_restarts
        self.pixle_max_iterations = pixle_max_iterations

        # VNIFGSM参数
        self.vni_decay = vni_decay
        self.vni_num_neighbor = vni_n
        self.vni_beta = vni_beta

        # 阶段权重 - Pixle优先
        self.pixle_ratio = pixle_ratio
        self.vni_ratio = vni_ratio

        # 计算各阶段迭代次数
        self.pixle_steps = int(self.epoch * self.pixle_ratio)
        self.vni_steps = self.epoch - self.pixle_steps

        logging.info(
            f"Initialized Pixle-Prioritized Hybrid with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, total_steps={self.epoch}, "
            f"pixle_steps={self.pixle_steps}, vni_steps={self.vni_steps}, "
            f"pixle_dims=({self.pixle_x_dimensions},{self.pixle_y_dimensions}), "
            f"pixle_restarts={self.pixle_restarts}, pixle_max_iter={self.pixle_max_iterations}"
        )
        logging.info("Strategy: Pixle-first (80%) for high platform score + VNIFGSM (20%) for SSIM optimization")

        # 初始化Pixle攻击 - 使用最优参数
        self.pixle_attack = torchattacks.Pixle(
            model,
            x_dimensions=self.pixle_x_dimensions,
            y_dimensions=self.pixle_y_dimensions,
            pixel_mapping=self.pixle_pixel_mapping,
            restarts=self.pixle_restarts,
            max_iterations=self.pixle_max_iterations,
            update_each_iteration=False
        )

        # 初始化VNIFGSM攻击
        dummy_model_name = 'resnet18'
        self.vni_attack = TransferAttackVNIFGSM(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            beta=self.vni_beta,
            num_neighbor=self.vni_num_neighbor,
            epoch=self.vni_steps,  # 只用于VNIFGSM阶段
            decay=self.vni_decay,
            targeted=False,
            random_start=False,  # 从Pixle结果开始，不需要随机起点
            norm='linfty',
            loss='crossentropy',
            device=self.device
        )

        # 用真实模型覆盖
        self.vni_attack.model = self.model
        logging.info("Both Pixle and VNIFGSM components initialized successfully.")

    def attack(self, images, labels):
        """
        执行两阶段混合攻击：
        阶段1: Pixle主攻击（高比例）
        阶段2: VNIFGSM微调优化
        """
        batch_size = images.shape[0]

        # 阶段1: Pixle主攻击
        logging.info(f"Stage 1: Pixle main attack for {self.pixle_steps} iterations")
        current_adv = images.clone()

        for pixle_step in range(self.pixle_steps):
            # 执行Pixle攻击
            pixle_adv = self.pixle_attack(current_adv, labels)

            # 确保扰动在epsilon范围内
            delta = pixle_adv - images
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            current_adv = torch.clamp(images + delta, 0, 1)

        stage1_adv = current_adv

        # 阶段2: VNIFGSM微调优化
        logging.info(f"Stage 2: VNIFGSM fine-tuning for {self.vni_steps} steps")

        # 使用Pixle的结果作为VNIFGSM的起点
        vni_perturbation = self.vni_attack(stage1_adv, labels)
        final_adv = stage1_adv + vni_perturbation
        final_adv = torch.clamp(final_adv, 0, 1)

        # 最终效果评估
        with torch.no_grad():
            final_outputs = self.model(final_adv)
            final_success = (final_outputs.argmax(1) != labels).float().mean().item()

        logging.info(f"Final hybrid attack success rate: {final_success:.4f}")

        return final_adv

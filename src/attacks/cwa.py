# src/attacks/cwa.py

import logging
import torch
from .base_attack import BaseAttack

try:
    from transferattack.ensemble.cwa import CWA as TransferAttackCWA
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'CWA' from 'transferattack.gradient.cwa' package."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class CWA(BaseAttack):
    def __init__(self, model, eps, alpha, steps, decay=1.0, beta=50, r_size=16/255/15, inner_step_size=250):
        """
        Initializes the TransferAttack AdaEA wrapper.
        """
        logging.info("Initializing TransferAttack CWA Wrapper...")
        super().__init__(model)

        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps
        self.decay = decay
        self.beta = beta
        self.r_size = r_size
        self.inner_step_size = inner_step_size

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        # 获取实际的模型数量
        if hasattr(model, 'models'):
            self.num_models = len(model.models)
        elif hasattr(model, 'num_model'):
            self.num_models = model.num_model
        else:
            self.num_models = 1

        logging.info(
            f"Initialized TransferAttack CWA with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, beta={self.beta}, r_size={self.r_size:.4f}, inner_step_size={self.inner_step_size:.4f}, "
            f"num_models={self.num_models}"
        )

        # 使用第一个模型名称作为占位符
        first_model_name = 'resnet18'

        # 创建TransferAttack的AdaEA实例
        self.attack_fn = TransferAttackCWA(
            model_name=first_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            epoch=self.epoch,
            decay=self.decay,
            beta=self.beta,
            r_size=self.r_size,
            inner_step_size=self.inner_step_size,
            targeted=False,
            random_start=True,
            norm='linfty',
            loss='crossentropy',
            device=self.device
        )

        # 用实际的模型替换内部的模型
        self.attack_fn.model = self.model
        # # 手动设置正确的模型数量,记得看原文件中模型数量的参数是什么（K要改）
        self.attack_fn.K = self.num_models

        # 确保模型处于评估模式
        self.attack_fn.model.eval()

        logging.info("TransferAttack's internal model was successfully overwritten with the framework's model.")

    def attack(self, images, labels):
        """
        Performs the attack using the wrapped TransferAttack object.

        Based on the TransferAttack library structure, the attack instance
        should be callable and return the perturbation (delta).
        """
        # 调用攻击实例，获取扰动
        perturbation = self.attack_fn(images, labels)

        # 将扰动加到原始图像上
        adv_images = images + perturbation

        # 确保对抗样本在有效范围内 [0, 1]
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images
# src/attacks/aifgtm.py

import logging
import torch
from .base_attack import BaseAttack

try:
    from transferattack.gradient.aifgtm import AIFGTM as TransferAttackAIFGTM
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'AIFGTM' from 'transferattack.gradient.aifgtm' package."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class AIFGTM(BaseAttack):
    """
    Wrapper for the AIFGTM attack from the 'TransferAttack' library (AAAI 2022).
    [https://arxiv.org/abs/2007.03838]

    AI-FGTM: Making Adversarial Examples More Transferable and Indistinguishable

    This class adapts the TransferAttack library's interface to
    fit this project's BaseAttack abstract class.
    """

    def __init__(self, model, eps, alpha, steps, decay=1.0, beta_1=0.9, beta_2=0.99, lam=1.3, mu_1=1.5, mu_2=1.9):
        """
        Initializes the TransferAttack AIFGTM wrapper.

        :param model: The model to attack (should be NormalizedModel and on the correct device).
        :param eps: Max perturbation budget (epsilon).
        :param alpha: Step size.
        :param steps: Number of attack iterations (TransferAttack calls this 'epoch').
        :param decay: Momentum decay factor.
        :param beta_1: First exponential decay rate for momentum.
        :param beta_2: Second exponential decay rate for variance.
        :param lam: Scale factor.
        :param mu_1: First decay factor.
        :param mu_2: Second decay factor.
        """
        logging.info("Initializing TransferAttack AIFGTM Wrapper...")
        super().__init__(model)

        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps  # 'steps' in our framework -> 'epoch' in TransferAttack
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lam = lam
        self.mu_1 = mu_1
        self.mu_2 = mu_2

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        logging.info(
            f"Initialized TransferAttack AIFGTM with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, beta_1={self.beta_1}, beta_2={self.beta_2}, "
            f"lam={self.lam}, mu_1={self.mu_1}, mu_2={self.mu_2}."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init

        # 创建TransferAttack的AIFGTM实例
        self.attack_fn = TransferAttackAIFGTM(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            epoch=self.epoch,
            decay=self.decay,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=self.device,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            lam=self.lam,
            mu_1=self.mu_1,
            mu_2=self.mu_2
        )

        # 用实际的模型替换内部的模型
        self.attack_fn.model = self.model
        # 确保模型处于评估模式
        self.attack_fn.model.eval()

        logging.info(
            "TransferAttack's internal model was successfully overwritten with the framework's model."
        )

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
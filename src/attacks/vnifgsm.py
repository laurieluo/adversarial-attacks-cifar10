# src/attacks/vnifgsm.py

import logging
import torch
from .base_attack import BaseAttack

try:
    from transferattack.gradient.vmifgsm import VMIFGSM
    from transferattack.gradient.vnifgsm import VNIFGSM as TransferAttackVNIFGSM
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'VNIFGSM' or 'VMIFGSM' from 'transferattack.gradient.*' package."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class VNIFGSM(BaseAttack):
    """
    Wrapper for the VNIFGSM attack from the 'TransferAttack' library (CVPR 2021).
    [https://arxiv.org/abs/2103.15571]
    
    This class adapts the TransferAttack library's interface to
    fit this project's BaseAttack abstract class.
    """

    # --- 关键修改 1: 在 __init__ 中接收所有参数 ---
    def __init__(self, model, eps, alpha, steps, decay=1.0, n=20, beta=1.5):
        """
        Initializes the TransferAttack VNIFGSM wrapper.

        :param model: The model to attack (should be NormalizedModel and on the correct device).
        :param eps: Max L-inf perturbation (TransferAttack calls this 'epsilon').
        :param alpha: Step size.
        :param steps: Number of attack iterations (TransferAttack calls this 'epoch').
        :param decay: Momentum decay factor.
        :param n: Number of neighbors (TransferAttack calls this 'num_neighbor').
        :param beta: Beta value for variance tuning.
        """
        logging.info("Initializing TransferAttack VNIFGSM Wrapper...")
        super().__init__(model)

        # --- 关键修改 2: 使用传入的参数，而不是硬编码 ---
        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps      # 'steps' in our framework -> 'epoch' in TransferAttack
        self.decay = decay
        self.num_neighbor = n   # 'n' in our framework -> 'num_neighbor' in TransferAttack
        self.beta = beta
        # --- 修改结束 ---

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        logging.info(
            f"Initialized TransferAttack VNIFGSM with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, num_neighbor={self.num_neighbor}, beta={self.beta}."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init
        
        self.attack_fn = TransferAttackVNIFGSM(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            beta=self.beta,
            num_neighbor=self.num_neighbor,
            epoch=self.epoch,
            decay=self.decay,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=self.device
        )

        # Overwrite the dummy model with the real one
        self.attack_fn.model = self.model
        self.attack_fn.model.eval()  
        
        logging.info(
            "TransferAttack's internal model was successfully overwritten with the framework's model."
        )

    def attack(self, images, labels):
        """
        Performs the attack using the wrapped TransferAttack object.
        
        Based on the source code, self.attack_fn() returns the
        perturbation (delta), NOT the final adversarial image.
        """
        
        perturbation = self.attack_fn(images, labels)
        adv_images_unclamped = images + perturbation
        adv_images_clamped = torch.clamp(adv_images_unclamped, 0, 1)
        
        return adv_images_clamped
# src/attacks/mef.py

import logging
import torch
from .base_attack import BaseAttack

try:
    # Attempt import from 'gradient' subdir
    from transferattack.gradient.mef import MEF as TransferAttackMEF
except ImportError:
    # Fallback import
    try:
        from transferattack.mef import MEF as TransferAttackMEF
    except ImportError as e:
        logging.error(
            "FATAL: Failed to import 'MEF' from 'transferattack.gradient.mef' or 'transferattack.mef'."
        )
        logging.error(
            "Ensure 'TransferAttack' repo is cloned in the project root "
            "and 'mef.py' exists in the correct path."
        )
        logging.error(f"Original error: {e}")
        raise e


class MEF(BaseAttack):
    """
    Wrapper for the MEF attack from the 'TransferAttack' library (2024).
    [https://arxiv.org/abs/2405.16181]
    
    This class adapts the TransferAttack library's interface to
    fit this project's BaseAttack abstract class.
    """

    def __init__(self, model, eps, alpha, steps, decay=0.5, inner_decay=0.9, n=20, gamma=2.0, kesai=0.15):
        """
        Initializes the TransferAttack MEF wrapper.

        :param model: The model to attack.
        :param eps: Max L-inf perturbation (epsilon).
        :param alpha: Step size.
        :param steps: Number of attack iterations (epoch).
        :param decay: Outer momentum decay factor.
        :param inner_decay: Inner momentum decay factor.
        :param n: Number of neighbors (num_neighbor).
        :param gamma: Upper bound of random sampling.
        :param kesai: Upper bound of sub-regions.
        """
        logging.info("Initializing TransferAttack MEF Wrapper...")
        super().__init__(model)

        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps      # 'steps' maps to 'epoch'
        self.decay = decay
        self.inner_decay = inner_decay
        self.num_neighbor = n   # 'n' maps to 'num_neighbor'
        self.gamma = gamma
        self.kesai = kesai

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        logging.info(
            f"Initialized TransferAttack MEF with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, inner_decay={self.inner_decay}, num_neighbor={self.num_neighbor}, "
            f"gamma={self.gamma}, kesai={self.kesai}."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init
        
        # Instantiate the real TransferAttack MEF attack
        self.attack_fn = TransferAttackMEF(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            num_neighbor=self.num_neighbor,
            gamma=self.gamma,
            kesai=self.kesai,
            epoch=self.epoch,
            inner_decay=self.inner_decay,
            decay=self.decay,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy_no_reduction', # Based on MEF source
            device=self.device
        )

        # Critical: Overwrite the dummy model with the real one
        self.attack_fn.model = self.model
        self.attack_fn.model.eval()  
        
        logging.info(
            "TransferAttack's internal model was successfully overwritten with the framework's model."
        )

    def attack(self, images, labels):
        """
        Performs the attack using the wrapped TransferAttack object.
        Returns perturbation (delta).
        """
        
        # Call TransferAttack's forward method, returns delta
        perturbation = self.attack_fn(images, labels)
        
        # Apply perturbation and clamp
        adv_images_unclamped = images + perturbation
        adv_images_clamped = torch.clamp(adv_images_unclamped, 0, 1)
        
        return adv_images_clamped
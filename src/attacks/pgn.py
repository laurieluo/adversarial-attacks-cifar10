# src/attacks/pgn.py

import logging
import torch
from .base_attack import BaseAttack

try:
    # Attempt import from 'gradient' subdir, consistent with vnifgsm
    from transferattack.gradient.pgn import PGN as TransferAttackPGN
except ImportError:
    # Fallback import: if PGN is in transferattack root
    try:
        from transferattack.pgn import PGN as TransferAttackPGN
    except ImportError as e:
        logging.error(
            "FATAL: Failed to import 'PGN' from 'transferattack.gradient.pgn' or 'transferattack.pgn'."
        )
        logging.error(
            "Ensure 'TransferAttack' repo is cloned in the project root "
            "and 'pgn.py' exists in the correct path."
        )
        logging.error(f"Original error: {e}")
        raise e


class PGN(BaseAttack):
    """
    Wrapper for the PGN attack from the 'TransferAttack' library (NeurIPS 2023).
    [https://arxiv.org/abs/2306.05225]
    
    This class adapts the TransferAttack library's interface to
    fit this project's BaseAttack abstract class.
    """

    def __init__(self, model, eps, alpha, steps, decay=1.0, beta=3.0, gamma=0.5, n=20):
        """
        Initializes the TransferAttack PGN wrapper.

        :param model: The model to attack.
        :param eps: Max L-inf perturbation (epsilon).
        :param alpha: Step size.
        :param steps: Number of attack iterations (epoch).
        :param decay: Momentum decay factor.
        :param beta: Relative value for the neighborhood.
        :param gamma: Balanced coefficient.
        :param n: Number of samples (num_neighbor).
        """
        logging.info("Initializing TransferAttack PGN Wrapper...")
        super().__init__(model)

        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps      # 'steps' maps to 'epoch'
        self.decay = decay
        self.beta = beta
        self.gamma = gamma
        self.num_neighbor = n   # 'n' maps to 'num_neighbor'

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        logging.info(
            f"Initialized TransferAttack PGN with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, beta={self.beta}, gamma={self.gamma}, num_neighbor={self.num_neighbor}."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init
        
        # Instantiate the real TransferAttack PGN attack
        self.attack_fn = TransferAttackPGN(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            num_neighbor=self.num_neighbor,
            epoch=self.epoch,
            decay=self.decay,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
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
        Based on the source code, self.attack_fn() returns the
        perturbation (delta), NOT the final adversarial image.
        """
        
        # Call TransferAttack's forward method, returns delta
        perturbation = self.attack_fn(images, labels)
        
        # Apply perturbation and clamp
        adv_images_unclamped = images + perturbation
        adv_images_clamped = torch.clamp(adv_images_unclamped, 0, 1)
        
        return adv_images_clamped
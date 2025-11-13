# src/attacks/ops.py

import logging
import numpy as np
import torch
from .base_attack import BaseAttack

try:
    from transferattack.input_transformation.ops import OPS as TransferAttackOPS
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'OPS' from 'transferattack.input_transformation.ops' package."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class OPS(BaseAttack):
    """
    Wrapper for the OPS attack from the 'TransferAttack' library (CVPR 2025).
    [Boosting Adversarial Transferability through Augmentation in Hypothesis Space]
    
    This class adapts the TransferAttack library's interface to
    fit this project's BaseAttack abstract class.
    
    **Dataset Differences (ImageNet vs CIFAR-10):**
    
    OPS was originally designed for ImageNet (224x224). When used on CIFAR-10 (32x32):
    
    1. **Image Size Impact**: Smaller images (32x32) may be more sensitive to:
       - Large rotation angles (45°, 90°, 180°) can significantly alter content
       - Aggressive scaling operations (scaling 2-8) may be too strong
       - Large resize_rate in dim operations (2.1-2.9) may cause artifacts
       
    2. **Recommended Adjustments for CIFAR-10**:
       - Use smaller epsilon (8/255 instead of 16/255) for better balance
       - Consider reducing num_sample_operator for faster execution
       - The default parameters should still work, but may be suboptimal
       
    3. **Known Issues**:
       - vertical_shift/horizontal_shift functions have variable naming issues
         but should still function correctly for both datasets
       - dim operations with large resize_rate may cause more artifacts on small images
    """

    def __init__(self, model, eps=16/255, alpha=None, steps=10, decay=1.0, 
                 beta=2., num_sample_neighbor=10, num_sample_operator=20, 
                 sample_levels=None, sample_ratios=None):
        """
        Initializes the TransferAttack OPS wrapper.

        :param model: The model to attack (should be NormalizedModel and on the correct device).
        :param eps: Max L-inf perturbation (TransferAttack calls this 'epsilon').
                    For CIFAR-10, consider using 8/255 instead of 16/255.
        :param alpha: Step size. If None, will be set to eps/steps.
        :param steps: Number of attack iterations (TransferAttack calls this 'epoch').
        :param decay: Momentum decay factor.
        :param beta: Beta value for perturbation sampling.
        :param num_sample_neighbor: Number of neighbor samples for perturbation sampling.
        :param num_sample_operator: Number of operator samples for operator sampling.
                                    For CIFAR-10, consider reducing to 10-15 for faster execution.
        :param sample_levels: Levels for operator sampling. Default: range(2, 5).
        :param sample_ratios: Ratios for perturbation sampling. Default: np.arange(0., 1.5, 0.25) + 0.25.
        """
        logging.info("Initializing TransferAttack OPS Wrapper...")
        super().__init__(model)

        self.epsilon = eps
        self.alpha = alpha if alpha is not None else eps / steps
        self.epoch = steps      # 'steps' in our framework -> 'epoch' in TransferAttack
        self.decay = decay
        self.beta = beta
        self.num_sample_neighbor = num_sample_neighbor
        self.num_sample_operator = num_sample_operator
        self.sample_levels = sample_levels if sample_levels is not None else range(2, 5)
        self.sample_ratios = sample_ratios if sample_ratios is not None else (np.arange(0., 1.5, 0.25) + 0.25)

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        # Detect image size for logging
        # Note: We can't check actual image size here, but we can log the parameters
        logging.info(
            f"Initialized TransferAttack OPS with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, beta={self.beta}, "
            f"num_sample_neighbor={self.num_sample_neighbor}, num_sample_operator={self.num_sample_operator}."
        )
        logging.info(
            f"Note: OPS was designed for ImageNet (224x224). For CIFAR-10 (32x32), "
            f"consider using eps=8/255 and reducing num_sample_operator for better performance."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init
        
        self.attack_fn = TransferAttackOPS(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            beta=self.beta,
            epoch=self.epoch,
            num_sample_neighbor=self.num_sample_neighbor,
            num_sample_operator=self.num_sample_operator,
            sample_levels=self.sample_levels,
            sample_ratios=self.sample_ratios,
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


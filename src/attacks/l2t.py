# src/attacks/l2t.py

import logging
import torch
from .base_attack import BaseAttack

try:
    from transferattack.input_transformation.l2t import L2T as TransferAttackL2T
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'L2T' from 'transferattack.input_transformation.l2t' package."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class L2T(BaseAttack):
    """
    Wrapper for the L2T attack from the 'TransferAttack' library.
    [Learning to Transform Dynamically for Better Adversarial Transferability]
    
    This class adapts the TransferAttack library's interface to
    fit this project's BaseAttack abstract class.
    
    **Dataset Differences (ImageNet vs CIFAR-10):**
    
    L2T was originally designed for ImageNet (224x224). When used on CIFAR-10 (32x32):
    
    1. **Image Size Impact**: Smaller images (32x32) may be more sensitive to:
       - Some operations in op_list may have hardcoded ImageNet dimensions (e.g., ssm class)
       - Large transformations may cause more artifacts on small images
       - The dynamic learning mechanism should adapt, but may be suboptimal
       
    2. **Recommended Adjustments for CIFAR-10**:
       - Use smaller epsilon (8/255 instead of 16/255) for better balance
       - Consider reducing num_scale for faster execution
       - The default parameters should still work, but may be suboptimal
       
    3. **Known Issues**:
       - Some operations (like ssm) may have hardcoded 224x224 dimensions
       - The attack may still work but with potential performance degradation
    """

    def __init__(self, model, eps=16/255, alpha=None, steps=10, decay=1.0, num_scale=3):
        """
        Initializes the TransferAttack L2T wrapper.

        :param model: The model to attack (should be NormalizedModel and on the correct device).
        :param eps: Max L-inf perturbation (TransferAttack calls this 'epsilon').
                    For CIFAR-10, consider using 8/255 instead of 16/255.
        :param alpha: Step size. If None, will be set to eps/steps.
        :param steps: Number of attack iterations (TransferAttack calls this 'epoch').
        :param decay: Momentum decay factor.
        :param num_scale: Number of scales for input transformation. Default: 3.
        """
        logging.info("Initializing TransferAttack L2T Wrapper...")
        super().__init__(model)

        self.epsilon = eps
        self.alpha = alpha if alpha is not None else eps / steps
        self.epoch = steps      # 'steps' in our framework -> 'epoch' in TransferAttack
        self.decay = decay
        self.num_scale = num_scale

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        logging.info(
            f"Initialized TransferAttack L2T with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, num_scale={self.num_scale}."
        )
        logging.info(
            f"Note: L2T was designed for ImageNet (224x224). For CIFAR-10 (32x32), "
            f"consider using eps=8/255 for better performance."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init
        
        self.attack_fn = TransferAttackL2T(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            epoch=self.epoch,
            decay=self.decay,
            num_scale=self.num_scale,
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


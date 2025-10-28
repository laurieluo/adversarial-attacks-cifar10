# src/attacks/vnifgsm.py
import torchattacks
from .base_attack import BaseAttack

class VNIFGSM(BaseAttack):
    """
    Implements the VNIFGSM attack (L-inf).
    'Enhancing the Transferability of Adversarial Attacks through Variance Tuning'
    [https://arxiv.org/abs/2103.15571]
    
    This is a momentum-based iterative attack that uses variance tuning.
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param eps: Max L-inf perturbation.
        :param alpha: Step size.
        :param steps: Number of attack iterations.
        :param decay: Momentum decay factor.
        """
        super().__init__(model)
        print(f"[VNIFGSM Attack] Initializing with eps={eps:.4f}, alpha={alpha:.4f}, steps={steps}")
        
        self.attack_fn = torchattacks.VNIFGSM(
            model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            decay=decay
        )

    def attack(self, images, labels):
        """
        Generates VNIFGSM adversarial examples.
        """
        return self.attack_fn(images, labels)
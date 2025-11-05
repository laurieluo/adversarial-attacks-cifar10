import torchattacks
import logging
from .base_attack import BaseAttack

class BIM(BaseAttack):
    """
    Implements the Basic Iterative Method (BIM) attack
    (also known as I-FGSM).
    
    This is essentially PGD without the random start.
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param eps: L-infinity norm (max perturbation).
        :param alpha: Step size per iteration.
        :param steps: Number of attack iterations.
        """
        logging.info("Initializing BIM attack...")

        super().__init__(model)
        
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = False # BIM is PGD without random start

        # Initialize the torchattacks BIM object
        # Note: BIM is PGD without random_start
        self.attack_fn = torchattacks.BIM(
            model, 
            eps=self.eps, 
            alpha=self.alpha, 
            steps=self.steps
        )
        
        logging.info(
            f"Initialized BIM (I-FGSM) Attack: "
            f"eps={self.eps:.4f}, alpha={self.alpha:.4f}, steps={self.steps}, "
            f"random_start={self.random_start}"
        )

    def attack(self, images, labels):
        """
        Generates BIM adversarial examples.
        """
        return self.attack_fn(images, labels)
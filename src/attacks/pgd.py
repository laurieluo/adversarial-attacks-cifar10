import torchattacks
import logging
from .base_attack import BaseAttack

class PGD(BaseAttack):
    """
    Implements the Projected Gradient Descent (PGD) attack
    using the torchattacks library.
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param eps: L-infinity norm (max perturbation).
        :param alpha: Step size per iteration.
        :param steps: Number of attack iterations.
        """
        logging.info("Initializing PGD attack...")

        super().__init__(model)
        
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

        self.attack_fn = torchattacks.PGD(
            model, 
            eps=self.eps, 
            alpha=self.alpha, 
            steps=self.steps, 
            random_start=self.random_start
        )
        
        logging.info(
            f"Initialized PGD Attack: "
            f"eps={self.eps:.4f}, alpha={self.alpha:.4f}, steps={self.steps}, "
            f"random_start={self.random_start}"
        )

    def attack(self, images, labels):
        """
        Generates PGD adversarial examples.
        """
        return self.attack_fn(images, labels)
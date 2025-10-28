# src/attacks/pgd.py
import torchattacks
from .base_attack import BaseAttack

class PGD(BaseAttack):
    """
    Implements the Projected Gradient Descent (PGD) attack
    using the torchattacks library.
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param eps: L-infinity norm (max perturbation).
        :param alpha: Step size per iteration.
        :param steps: Number of attack iterations.
        """
        super().__init__(model)
        self.attack_fn = torchattacks.PGD(
            model, 
            eps=eps, 
            alpha=alpha, 
            steps=steps, 
            random_start=True
        )

    def attack(self, images, labels):
        """
        Generates PGD adversarial examples.
        """
        return self.attack_fn(images, labels)

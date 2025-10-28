# src/attacks/bim.py
import torchattacks
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
        super().__init__(model)
        # Initialize the torchattacks BIM object
        # Note: BIM is PGD without random_start
        self.attack_fn = torchattacks.BIM(
            model, 
            eps=eps, 
            alpha=alpha, 
            steps=steps
        )

    def attack(self, images, labels):
        """
        Generates BIM adversarial examples.
        """
        return self.attack_fn(images, labels)
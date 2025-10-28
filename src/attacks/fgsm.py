# src/attacks/fgsm.py
import torchattacks
from .base_attack import BaseAttack

class FGSM(BaseAttack):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack
    using the torchattacks library.
    """
    def __init__(self, model, eps=8/255):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param eps: L-infinity norm (max perturbation).
        """
        super().__init__(model)
        # Initialize the torchattacks FGSM object
        self.attack_fn = torchattacks.FGSM(model, eps=eps)

    def attack(self, images, labels):
        """
        Generates FGSM adversarial examples.
        """
        # Pass images and labels to the attack function
        return self.attack_fn(images, labels)

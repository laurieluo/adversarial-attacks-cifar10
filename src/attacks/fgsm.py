import torchattacks
import logging 
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
        logging.info("Initializing FGSM attack...")

        super().__init__(model)
        
        self.eps = eps
        
        # Initialize the torchattacks FGSM object
        self.attack_fn = torchattacks.FGSM(model, eps=self.eps)

        logging.info(
            f"Initialized FGSM Attack: "
            f"eps={self.eps:.4f}"
        )

    def attack(self, images, labels):
        """
        Generates FGSM adversarial examples.
        """
        # Pass images and labels to the attack function
        return self.attack_fn(images, labels)
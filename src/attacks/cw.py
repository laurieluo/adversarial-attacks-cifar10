import torchattacks
import logging
from .base_attack import BaseAttack

class CW(BaseAttack):
    """
    Implements the Carlini & Wagner (C&W) L2 attack.
    Warning: This attack is very slow.
    """
    def __init__(self, model, c=1, kappa=0, steps=1000, lr=0.01):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param c: Binary search steps (controls the trade-off)
        :param kappa: Confidence of the attack (0=default)
        :param steps: Max steps for the optimization
        :param lr: Learning rate for the optimization
        """
        logging.info("Initializing CW attack...")

        super().__init__(model)
        
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        
        # Initialize the torchattacks CW object
        self.attack_fn = torchattacks.CW(
            model, 
            c=self.c, 
            kappa=self.kappa, 
            steps=self.steps, 
            lr=self.lr
        )
        
        logging.info(
            f"Initialized CW Attack: "
            f"c={self.c}, kappa={self.kappa}, steps={self.steps}, lr={self.lr}"
        )
        logging.warning(
            "CW Attack is extremely slow. This may take several minutes or hours."
        )


    def attack(self, images, labels):
        """
        Generates CW adversarial examples.
        """
        return self.attack_fn(images, labels)
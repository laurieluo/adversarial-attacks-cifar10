# src/attacks/cw.py
import torchattacks
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
        super().__init__(model)
        # Initialize the torchattacks CW object
        self.attack_fn = torchattacks.CW(
            model, 
            c=c, 
            kappa=kappa, 
            steps=steps, 
            lr=lr
        )

    def attack(self, images, labels):
        """
        Generates CW adversarial examples.
        """
        return self.attack_fn(images, labels)
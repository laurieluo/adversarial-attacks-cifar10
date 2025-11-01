# src/attacks/onepixel.py
import torchattacks
from .base_attack import BaseAttack

class OnePixel(BaseAttack):
    """
    Implements the One-Pixel Attack (L0).
    'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]

    This attack modifies only one pixel to create an adversarial example.
    It uses Differential Evolution (DE) and is very slow.
    """
    def __init__(self, model, pixels=1, steps=10, popsize=10, inf_batch=128):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param pixels: Number of pixels to change. (Default: 1)
        :param steps: Number of steps (iterations) for DE. (Default: 10)
        :param popsize: Population size for DE. (Default: 10)
        :param inf_batch: Max batch size during inference. (Default: 128)
        """
        super().__init__(model)
        
        self.attack_fn = torchattacks.OnePixel(
            model,
            pixels=pixels,
            steps=steps,
            popsize=popsize,
            inf_batch=inf_batch
        )

    def attack(self, images, labels):
        """
        Generates OnePixel adversarial examples.
        """
        return self.attack_fn(images, labels)
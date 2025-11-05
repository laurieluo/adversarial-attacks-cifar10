import torchattacks
import logging  # 1. Import the logging module
from .base_attack import BaseAttack

class OnePixel(BaseAttack):
    """
    Implements the One-Pixel Attack (L0).
    'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]

    This attack modifies only one (or few) pixel(s) to create an adversarial example.
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
        logging.info("Initializing OnePixel Attack...")

        super().__init__(model)
        
        # 2. Store parameters
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        self.inf_batch = inf_batch
        
        # 3. Log the initialization
        logging.info(
            f"Initialized OnePixel Attack: "
            f"pixels={self.pixels}, steps={self.steps}, "
            f"popsize={self.popsize}, inf_batch={self.inf_batch}"
        )
        logging.warning(
            "OnePixel Attack uses Differential Evolution and is EXTREMELY slow."
        )
        
        self.attack_fn = torchattacks.OnePixel(
            model,
            pixels=self.pixels,
            steps=self.steps,
            popsize=self.popsize,
            inf_batch=self.inf_batch
        )

    def attack(self, images, labels):
        """
        Generates OnePixel adversarial examples.
        """
        return self.attack_fn(images, labels)
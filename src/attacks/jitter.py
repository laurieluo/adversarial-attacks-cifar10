import torchattacks
import logging  # 1. Import the logging module
from .base_attack import BaseAttack

class Jitter(BaseAttack):
    """
    Implements the Jitter Attack (L-inf).
    'Exploring Misclassifications of Robust Neural Networks to Enhance Adversarial Attacks'
    [https://arxiv.org/abs/2105.10304]

    This is a PGD-based attack that applies random resizing and Gaussian noise
    at each iteration to find more robust gradients.
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, 
                 scale=10, std=0.1, random_start=True):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param eps: Maximum L-inf perturbation.
        :param alpha: Step size.
        :param steps: Number of attack iterations.
        :param scale: Controls the range of random resizing.
        :param std: Standard deviation of Gaussian noise.
        :param random_start: Using random initialization of delta.
        """
        logging.info("Initializing Jitter Attack...")

        super().__init__(model)
        
        # 2. Store parameters
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.scale = scale
        self.std = std
        self.random_start = random_start
        
        # 3. Log the initialization
        logging.info(
            f"Initialized Jitter Attack: "
            f"eps={self.eps:.4f}, alpha={self.alpha:.4f}, steps={self.steps}, "
            f"scale={self.scale}, std={self.std}, random_start={self.random_start}"
        )
        
        # Initialize the torchattacks Jitter object
        self.attack_fn = torchattacks.Jitter(
            model,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            scale=self.scale,
            std=self.std,
            random_start=self.random_start
        )

    def attack(self, images, labels):
        """
        Generates Jitter adversarial examples.
        """
        return self.attack_fn(images, labels)
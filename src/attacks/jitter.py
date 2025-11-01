# src/attacks/jitter.py
import torchattacks
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
        super().__init__(model)
        print(f"[Jitter Attack] Initializing with eps={eps:.4f}, alpha={alpha:.4f}, steps={steps}")
        
        # 使用您提供的文档中的正确参数
        self.attack_fn = torchattacks.Jitter(
            model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            scale=scale,
            std=std,
            random_start=random_start
        )

    def attack(self, images, labels):
        """
        Generates Jitter adversarial examples.
        """
        return self.attack_fn(images, labels)
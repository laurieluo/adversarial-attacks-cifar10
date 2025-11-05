import torchattacks
import logging  # 1. Import the logging module
from .base_attack import BaseAttack

class VNIFGSM(BaseAttack):
    """
    Implements the VNIFGSM attack (L-inf).
    'Enhancing the Transferability of Adversarial Attacks through Variance Tuning'
    [https://arxiv.org/abs/2103.15571]
    
    This is a momentum-based iterative attack that uses variance tuning.
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, n=5, beta=1.5):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param eps: Max L-inf perturbation.
        :param alpha: Step size.
        :param steps: Number of attack iterations.
        :param decay: Momentum decay factor.
        """
        logging.info("Initializing VNIFGSM Attack...")

        super().__init__(model)

        # 2. Store parameters
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.n = n
        self.beta = beta
        
        # 3. Log the initialization
        logging.info(
            f"Initialized VNIFGSM Attack: "
            f"eps={self.eps:.4f}, alpha={self.alpha:.4f}, steps={self.steps}, "
            f"decay={self.decay}, n={self.n}, beta={self.beta}."
        )
        
        self.attack_fn = torchattacks.VNIFGSM(
            model,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            decay=self.decay,
            N=self.n,
            beta=self.beta
        )

    def attack(self, images, labels):
        """
        Generates VNIFGSM adversarial examples.
        """
        return self.attack_fn(images, labels)
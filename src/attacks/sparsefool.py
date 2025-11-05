import torchattacks
import logging  # 1. Import the logging module
from .base_attack import BaseAttack

class SparseFool(BaseAttack):
    """
    Implements the SparseFool Attack (L0).
    'SparseFool: a few pixels make a big difference'
    [https://arxiv.org/abs/1811.02248]

    This attack finds a sparse (L0) adversarial perturbation.
    It is an iterative attack based on DeepFool and can be slow.
    """
    def __init__(self, model, steps=10, lam=3, overshoot=0.02):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param steps: Number of steps. (Default: 10)
        :param lam: Parameter for scaling DeepFool noise. (Default: 3)
        :param overshoot: Parameter for enhancing the noise. (Default: 0.02)
        """
        logging.info("Initializing SparseFool Attack...")

        super().__init__(model)
        
        # 2. Store parameters
        self.steps = steps
        self.lam = lam
        self.overshoot = overshoot
        
        # 3. Log the initialization
        logging.info(
            f"Initialized SparseFool Attack: "
            f"steps={self.steps}, lam={self.lam}, overshoot={self.overshoot}"
        )
        logging.warning(
            "SparseFool Attack is iterative and can be slow."
        )
        
        # Initialize the torchattacks SparseFool object
        self.attack_fn = torchattacks.SparseFool(
            model,
            steps=self.steps,
            lam=self.lam,
            overshoot=self.overshoot
        )

    def attack(self, images, labels):
        """
        Generates SparseFool adversarial examples.
        
        Note: SparseFool is a non-targeted attack by default
        and does not use the 'labels' parameter in its main logic,
        as it tries to find the closest decision boundary.
        The torchattacks library implementation still requires 'labels' 
        in the forward pass signature.
        """
        # Pass images and labels to the attack function
        return self.attack_fn(images, labels)
import torchattacks
import logging  # 1. Import the logging module
from .base_attack import BaseAttack

class AutoAttack(BaseAttack):
    """
    Implements the AutoAttack (AA) benchmark.
    AA is a combination of four strong attacks.
    Warning: This attack is significantly slower than PGD.
    """
    def __init__(self, model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param norm: 'Linf' or 'L2' (default: 'Linf')
        :param eps: Max perturbation (default: 8/255)
        """
        logging.info("Initializing AutoAttack...")

        super().__init__(model)
        
        # 2. Store parameters
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        
        # 3. Log the initialization
        logging.info(
            f"Initialized AutoAttack (AA): "
            f"norm={self.norm}, eps={self.eps:.4f}, version={self.version}, n_classes={self.n_classes}, "
            f"seed={self.seed}, verbose={self.verbose}."
        )
        logging.warning(
            "AutoAttack is very slow. This may take several minutes."
        )
        
        # Initialize the torchattacks AutoAttack object
        # We set verbose=False to avoid excessive printing during the attack loop
        self.attack_fn = torchattacks.AutoAttack(
            model, 
            norm=self.norm, 
            eps=self.eps,
            version=self.version,
            n_classes=self.n_classes,
            seed=self.seed,
            verbose=self.verbose 
        )

    def attack(self, images, labels):
        """
        Generates AutoAttack adversarial examples.
        """
        return self.attack_fn(images, labels)
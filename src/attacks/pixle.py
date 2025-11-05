import torchattacks
import logging  # 1. Import the logging module
from .base_attack import BaseAttack

class Pixle(BaseAttack):
    """
    Implements the Pixle Attack (L0) based on pixel rearrangement.
    [https://arxiv.org/abs/2202.02236]
    This attack uses an optimization algorithm and can be slow.
    """
    def __init__(self, model, x_dimensions=(2, 10), y_dimensions=(2, 10), 
                 pixel_mapping='random', restarts=20, max_iterations=10):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param x_dimensions: Size/range of the patch x-side. (Default: (2, 10))
        :param y_dimensions: Size/range of the patch y-side. (Default: (2, 10))
        :param pixel_mapping: Type of mapping. (Default: 'random')
        :param restarts: Number of restarts. (Default: 20)
        :param max_iterations: Iterations per restart. (Default: 10)
        """
        logging.info("Initializing Pixle Attack...")

        super().__init__(model)
        
        # 2. Store parameters
        self.x_dimensions = x_dimensions
        self.y_dimensions = y_dimensions
        self.pixel_mapping = pixel_mapping
        self.restarts = restarts
        self.max_iterations = max_iterations
        
        # 3. Log the initialization
        logging.info(
            f"Initialized Pixle Attack: "
            f"x_dimensions={self.x_dimensions}, y_dimensions={self.y_dimensions}, "
            f"restarts={self.restarts}, max_iterations={self.max_iterations}, "
            f"pixel_mapping='{self.pixel_mapping}'"
        )
        logging.warning(
            "Pixle Attack uses an optimization algorithm and can be slow."
        )

        self.attack_fn = torchattacks.Pixle(
            model,
            x_dimensions=self.x_dimensions,
            y_dimensions=self.y_dimensions,
            pixel_mapping=self.pixel_mapping,
            restarts=self.restarts,
            max_iterations=self.max_iterations,
            update_each_iteration=False
        )

    def attack(self, images, labels):
        """
        Generates Pixle adversarial examples.
        """
        return self.attack_fn(images, labels)
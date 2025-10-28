# src/attacks/pixle.py
import torchattacks
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
        super().__init__(model)
        print(f"[Pixle Attack] Initializing with restarts={restarts}, max_iterations={max_iterations}")
        
        self.attack_fn = torchattacks.Pixle(
            model,
            x_dimensions=x_dimensions,
            y_dimensions=y_dimensions,
            pixel_mapping=pixel_mapping,
            restarts=restarts,
            max_iterations=max_iterations,
            update_each_iteration=False
        )

    def attack(self, images, labels):
        """
        Generates Pixle adversarial examples.
        """
        return self.attack_fn(images, labels)
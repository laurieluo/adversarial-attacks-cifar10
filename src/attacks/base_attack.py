# src/attacks/base_attack.py
from abc import ABC, abstractmethod

class BaseAttack(ABC):
    """
    Abstract Base Class (Interface) for all attack algorithms.
    
    This ensures that any new attack you add will have a consistent
    `.attack()` method.
    """
    def __init__(self, model):
        """
        Initializes the attack.
        :param model: The model to attack (should be pre-wrapped with NormalizedModel).
        """
        self.model = model

    @abstractmethod
    def attack(self, images, labels):
        """
        Performs the attack.
        :param images: A batch of original, clean images (Tensor) in [0, 1] range.
        :param labels: The true labels for these images (Tensor).
        :return: A batch of adversarial images (Tensor).
        """
        pass
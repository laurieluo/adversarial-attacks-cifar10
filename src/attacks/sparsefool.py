# src/attacks/sparsefool.py
import torchattacks
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
        super().__init__(model)
        print(f"[SparseFool Attack] Initializing with steps={steps}, lam={lam}, overshoot={overshoot}")
        
        # 使用您提供的文档中的正确参数
        self.attack_fn = torchattacks.SparseFool(
            model,
            steps=steps,
            lam=lam,
            overshoot=overshoot
        )

    def attack(self, images, labels):
        """
        Generates SparseFool adversarial examples.
        
        Note: SparseFool is a non-targeted attack by default
        and does not use the 'labels' parameter in its main logic,
        as it tries to find the closest decision boundary.
        """
        # SparseFool (torchattacks.SparseFool) 的 forward 方法
        return self.attack_fn(images, labels)
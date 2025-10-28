# src/attacks/autoattack.py
import torchattacks
from .base_attack import BaseAttack

class AutoAttack(BaseAttack):
    """
    Implements the AutoAttack (AA) benchmark.
    AA is a combination of four strong attacks.
    Warning: This attack is significantly slower than PGD.
    """
    def __init__(self, model, norm='Linf', eps=8/255):
        """
        :param model: The model to attack (should be NormalizedModel).
        :param norm: 'Linf' or 'L2' (default: 'Linf')
        :param eps: Max perturbation (default: 8/255)
        """
        super().__init__(model)
        print(f"[AutoAttack] Initializing with norm={norm}, eps={eps:.4f}")
        
        # 初始化 torchattacks 的 AutoAttack 对象
        # 我们设置 verbose=False 来避免在循环中打印过多信息
        self.attack_fn = torchattacks.AutoAttack(
            model, 
            norm=norm, 
            eps=eps,
            verbose=False 
        )

    def attack(self, images, labels):
        """
        Generates AutoAttack adversarial examples.
        """
        return self.attack_fn(images, labels)
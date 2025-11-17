# src/attacks/p2fa.py

import logging
import sys

import torch
import torchvision.transforms as T
from .base_attack import BaseAttack

try:
    from transferattack.gradient.mifgsm import MIFGSM
    from transferattack.advanced_objective.bfa import BFA as TransferAttackBFA
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'MIFGSM' or 'BFA'."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class BFA(BaseAttack):
    """
    BFA Attack
    Improving the transferability of adversarial examples through black-box feature attacks (Neurocomputing 2024) (https://www.sciencedirect.com/science/article/abs/pii/S0925231224006349)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        eta (float): the perturbation size for mask gradient.
        num_ens (int): the fitting iteration steps.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack
        drop_rate : probability to drop random pixel

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., eta=28, num_ens=30, layer_name='layer2.7' for ResNet152

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfa/resnet50 --attack bfa --model resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfa/resnet50 --eval

    NOTE:
        1) ResNet18 is not mentioned in the original paper. Following the setting for ResNet152 in the paper, we select the last block of the second layer for ResNet18 as the feature layer.
        2) The implementation refers to the official code of BFA attack (https://github.com/tlemangen/BFA).
    """

    def __init__(self, model, eps, alpha, steps, decay=1.0,  num_ens=30, layer_name= 'layer2', eta=28.0):
        """
        :param model: The model to attack (should be NormalizedModel and on the correct device).
        :param eps: Max L-inf perturbation (TransferAttack calls this 'epsilon').
        :param alpha: Step size.
        :param steps: Number of attack iterations (TransferAttack calls this 'epoch').
        :param decay: Momentum decay factor.
        :param num_ens: the number of gradients to aggregate.
        :param feature_layer: feature layer to launch the attack.
        """
        logging.info("Initializing TransferAttack BFA...")
        super().__init__(model)
        autual_layer_name = f"model.{layer_name}"

        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps  # 'steps' in our framework -> 'epoch' in TransferAttack
        self.decay = decay
        self.num_ens = num_ens
        self.layer_name = autual_layer_name
        self.feature_maps = None
        self.eta = eta

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        logging.info(
            f"Initialized TransferAttack BFA with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, decay={self.decay}, "
            f"ensemble_number={self.num_ens}, layer_name={self.layer_name}, eta={self.eta:.4f}."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init

        self.attack_fn = TransferAttackBFA(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            epoch=self.epoch,
            decay=self.decay,
            eta = self.eta,
            num_ens=self.num_ens,
            targeted=False,
            random_start=False,
            layer_name=self.layer_name,
            norm='linfty',
            loss='crossentropy',
            device=self.device
        )

        # Overwrite the dummy model with the real one
        self.attack_fn.model = self.model
        # 钩子在 init 中注册，此时 self.model 还未正确设置，因此需要手动注册钩子确保钩子注册到正确的模型
        self.attack_fn.register_hook()
        self.attack_fn.model.eval()

        logging.info(
            "TransferAttack's internal model was successfully overwritten with the framework's model."
        )

    def attack(self, images, labels):
        """
        Performs the attack using the wrapped TransferAttack object.

        Based on the source code, self.attack_fn() returns the
        perturbation (delta), NOT the final adversarial image.
        """
        # 以下为调试内容，经过调试发现模型层名称与预期不同，需要对名称再进行规范化
        # 手动实现钩子触发
        with torch.no_grad():
            _ = self.model(images[:1])  # 用第一个样本触发前向传播

        # # 如果特征图仍然为None，尝试重新注册钩子
        # if self.attack_fn.feature_maps is None:
        #     logging.warning("首次钩子触发失败，尝试重新注册钩子...")
        #     # 重新设置模型并注册钩子
        #     self.attack_fn.model = self.model
        #     self.attack_fn.register_hook()
        #
        #     # 再次尝试触发
        #     with torch.no_grad():
        #         _ = self.model(images[:1])

        # 最终检查
        if self.attack_fn.feature_maps is None:
            # 打印模型结构帮助调试
            layer_names = [name for name, _ in self.model.named_modules() if name]
            logging.error(f"模型层名称列表 (前20个): {layer_names[:20]}")
            logging.error("钩子仍未正确触发！特征图仍然是None")
            sys.exit(1)

        perturbation = self.attack_fn(images, labels)
        adv_images_unclamped = images + perturbation
        adv_images_clamped = torch.clamp(adv_images_unclamped, 0, 1)

        return adv_images_clamped
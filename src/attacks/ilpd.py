# src/attacks/ilpd.py

import logging
import sys

import torch
import torchvision.transforms as T
from .base_attack import BaseAttack

try:
    from transferattack.gradient.mifgsm import MIFGSM
    from transferattack.advanced_objective.ilpd import ILPD as TransferAttackILPD
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'MIFGSM' or 'ILPD'."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class ILPD(BaseAttack):
    """
    ILPD Attack
    'Improving Adversarial Transferability via Intermediate-level Perturbation Decay'(https://arxiv.org/abs/2304.13410)

    Arguments:
        attack (str): the name of attack.
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        coef (float): coeffcient gamma
        sigma (float): noise size
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epoch=100, sigma=0.05, coef=0.1, N=1, il_pos="layer2.3"

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ilpd/resnet50 --attack ilpd --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/ilpd/resnet50 --eval
    """

    def __init__(self, model, eps, alpha, steps, decay=1.0, il_module='layer2', sigma=0.05, coef=0.1, N=1):
        """
        :param model: The model to attack (should be NormalizedModel and on the correct device).
        :param eps: Max L-inf perturbation (TransferAttack calls this 'epsilon').
        :param alpha: Step size.
        :param steps: Number of attack iterations (TransferAttack calls this 'epoch').
        :param decay: Momentum decay factor.
        :param num_ens: the number of gradients to aggregate.
        :param feature_layer: feature layer to launch the attack.
        """
        logging.info("Initializing TransferAttack ILPD...")
        super().__init__(model)
        # 解析模块路径获取动态的模块对象
        self.il_module_path = f"model.{il_module}"
        self.actual_il_module = self.get_actual_il_module(model, self.il_module_path)

        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps  # 'steps' in our framework -> 'epoch' in TransferAttack
        self.decay = decay
        self.sigma = sigma
        self.coef = coef
        self.N = N

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        logging.info(
            f"Initialized TransferAttack ILPD with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, decay={self.decay}, "
            f"il_module={self.il_module_path}, sigma={self.sigma}, coef={self.coef:.4f}, N={self.N}."
        )

        dummy_model_name = 'resnet18'  # Placeholder for TransferAttack's init

        self.attack_fn = TransferAttackILPD(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            epoch=self.epoch,
            decay=self.decay,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=self.device,
            sigma=self.sigma,
            coef=self.coef,
            N=self.N
        )

        # Overwrite the dummy model with the real one
        self.attack_fn.model = self.model
        self.attack_fn.model.eval()

        # 由于原方法中硬编码了中间模块对象，且不接受参数传入，所以通过覆盖的方式设置实际的模块对象
        self.attack_fn.il_module = self.actual_il_module

        logging.info(
            "TransferAttack's internal model was successfully overwritten with the framework's model."
        )

    def get_actual_il_module(self, model, module_path):
        """
        从模型和模块路径中获取实际的模块对象
        支持复杂的模块路径，如 'model.features.15'
        """
        try:
            # 方法1: 使用PyTorch内置的get_submodule方法（推荐）
            if hasattr(model, 'get_submodule'):
                submodule = model.get_submodule(module_path)
                if not isinstance(submodule, torch.nn.Module):
                    logging.error(f"Resolved object is not a torch.nn.Module: {type(submodule)}")

                logging.info(f"Successfully resolved module path '{module_path}' to {type(submodule).__name__}")
                return submodule

            # 方法2: 手动解析路径（兼容旧版PyTorch）
            parts = module_path.split('.')
            current_module = model

            for part in parts:
                # 尝试直接获取属性
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                # 如果失败，尝试作为数字索引（对于Sequential模块）
                else:
                    try:
                        index = int(part)
                        if hasattr(current_module, '__getitem__'):
                            current_module = current_module[index]
                        else:
                            logging.error(f"Module '{current_module}' does not support indexing")
                    except ValueError:
                        logging.error(f"Module '{current_module}' has no attribute '{part}' and '{part}' is not a valid index")

            if not isinstance(current_module, torch.nn.Module):
                logging.error(f"Resolved object is not a torch.nn.Module: {type(current_module)}")

            logging.info(f"Successfully resolved module path '{module_path}' to {type(current_module).__name__}")
            return current_module

        except Exception as e:
            logging.error(f"Failed to resolve module path '{module_path}': {e}")

            # 打印可用的模块路径帮助调试
            logging.info("Available modules in model:")
            for name, module in model.named_modules():
                if name:  # 跳过空名字（根模块）
                    logging.info(f"  - {name}: {type(module).__name__}")

            sys.exit(1)

    def attack(self, images, labels):
        """
        Performs the attack using the wrapped TransferAttack object.

        Based on the source code, self.attack_fn() returns the
        perturbation (delta), NOT the final adversarial image.
        """

        # 手动注册钩子
        self.attack_fn.prep_hook(images)

        perturbation = self.attack_fn(images, labels)
        adv_images_unclamped = images + perturbation
        adv_images_clamped = torch.clamp(adv_images_unclamped, 0, 1)

        return adv_images_clamped
# src/attacks/vnifgsm_sim.py

import logging
import torch
import torch.nn.functional as F
from .base_attack import BaseAttack

try:
    from transferattack.gradient.vnifgsm import VNIFGSM as TransferAttackVNIFGSM
    from transferattack.input_transformation.sim import SIM as TransferAttackSIM
except ImportError as e:
    logging.error(
        "FATAL: Failed to import 'VNIFGSM' or 'SIM' from 'transferattack.*' package."
    )
    logging.error(
        "Ensure 'TransferAttack' repo is cloned in the project root "
        "and 'run_attack.py' adds it to sys.path."
    )
    logging.error(f"Original error: {e}")
    raise e


class VNIFGSM_SIM(BaseAttack):
    """
    高效的混合攻击算法：结合VNIFGSM的方差调优和动量特性与SIM的多尺度不变性。

    这种混合攻击充分利用了：
    - VNIFGSM：通过方差调优和Nesterov动量来稳定梯度方向
    - SIM：通过多尺度输入变换来增强攻击的迁移性
    """

    def __init__(self, model, eps, alpha, steps, decay=1.0, n=20, beta=1.5,
                 num_scale=5, scale_factor=1.1, momentum_weight=0.6, sim_weight=0.4):
        """
        初始化VNIFGSM-SIM混合攻击。

        :param model: 要攻击的模型（应该是NormalizedModel且在正确的设备上）
        :param eps: L∞扰动上限
        :param alpha: 步长
        :param steps: 攻击迭代次数
        :param decay: VNIFGSM的动量衰减因子
        :param n: VNIFGSM的邻居数量
        :param beta: VNIFGSM的方差调优参数
        :param num_scale: SIM的尺度数量
        :param scale_factor: SIM的最大缩放因子
        :param momentum_weight: VNIFGSM分量的权重
        :param sim_weight: SIM分量的权重
        """
        logging.info("Initializing VNIFGSM-SIM Hybrid Attack...")
        super().__init__(model)

        # 基础参数
        self.epsilon = eps
        self.alpha = alpha
        self.epoch = steps

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logging.warning("Could not infer device from model, defaulting to CPU.")

        # VNIFGSM参数
        self.decay = decay
        self.num_neighbor = n
        self.beta = beta

        # SIM参数
        self.num_scale = num_scale
        self.scale_factor = scale_factor

        # 混合权重
        self.momentum_weight = momentum_weight
        self.sim_weight = sim_weight

        # 权重归一化
        total_weight = momentum_weight + sim_weight
        self.momentum_weight /= total_weight
        self.sim_weight /= total_weight

        logging.info(
            f"Initialized VNIFGSM-SIM Hybrid with parameters: "
            f"epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, epoch={self.epoch}, "
            f"decay={self.decay}, num_neighbor={self.num_neighbor}, beta={self.beta}, "
            f"num_scale={self.num_scale}, scale_factor={self.scale_factor}, "
            f"momentum_weight={self.momentum_weight:.2f}, sim_weight={self.sim_weight:.2f}."
        )

        # 初始化两个攻击组件
        dummy_model_name = 'resnet18'

        self.vnifgsm_attack = TransferAttackVNIFGSM(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            beta=self.beta,
            num_neighbor=self.num_neighbor,
            epoch=self.epoch,
            decay=self.decay,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=self.device
        )

        self.sim_attack = TransferAttackSIM(
            model_name=dummy_model_name,
            epsilon=self.epsilon,
            alpha=self.alpha,
            epoch=self.epoch,
            num_scale=self.num_scale,
            scale_factor=self.scale_factor,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=self.device
        )

        # 用真实模型覆盖虚拟模型
        self.vnifgsm_attack.model = self.model
        self.sim_attack.model = self.model
        self.vnifgsm_attack.model.eval()
        self.sim_attack.model.eval()

        logging.info("Both VNIFGSM and SIM components initialized with framework's model.")

    def _compute_vnifgsm_gradient(self, images, labels, momentum):
        """计算VNIFGSM梯度（带方差调优）"""
        images = images.clone().detach().requires_grad_(True)

        # 前向传播
        outputs = self.model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # 计算梯度
        grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

        # 方差调优：在邻域内采样计算梯度方差
        grad_var = torch.zeros_like(grad)
        for _ in range(self.num_neighbor):
            # 在输入空间添加小扰动
            neighbor = images + torch.randn_like(images) * (self.epsilon * 0.1)
            neighbor = torch.clamp(neighbor, 0, 1).detach().requires_grad_(True)

            neighbor_outputs = self.model(neighbor)
            neighbor_loss = torch.nn.functional.cross_entropy(neighbor_outputs, labels)
            neighbor_grad = torch.autograd.grad(neighbor_loss, neighbor,
                                                retain_graph=False, create_graph=False)[0]
            grad_var += neighbor_grad

        grad_var = grad_var / self.num_neighbor

        # 结合当前梯度和梯度方差
        tuned_grad = grad + self.beta * (grad_var - grad)

        return tuned_grad.detach()

    def _compute_sim_gradient(self, images, labels):
        """计算SIM梯度（多尺度平均）"""
        batch_size = images.shape[0]
        total_grad = torch.zeros_like(images)

        # 生成多个尺度
        scales = [1.0]  # 包含原始尺度
        for i in range(1, self.num_scale):
            scale = 1.0 + (self.scale_factor - 1.0) * i / (self.num_scale - 1)
            scales.append(scale)

        for scale in scales:
            if scale == 1.0:
                # 原始尺度
                scaled_images = images.clone().detach().requires_grad_(True)
            else:
                # 缩放图像
                scaled_size = (int(images.shape[2] * scale), int(images.shape[3] * scale))
                scaled_images = F.interpolate(images, size=scaled_size, mode='bilinear',
                                              align_corners=False)
                scaled_images = F.interpolate(scaled_images, size=images.shape[2:],
                                              mode='bilinear', align_corners=False)
                scaled_images = torch.clamp(scaled_images, 0, 1)
                scaled_images = scaled_images.detach().requires_grad_(True)

            # 前向传播
            outputs = self.model(scaled_images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            # 计算梯度
            grad = torch.autograd.grad(loss, scaled_images, retain_graph=False, create_graph=False)[0]
            total_grad += grad

        # 平均梯度
        return total_grad / len(scales)

    def attack(self, images, labels):
        """
        执行混合攻击。

        策略：在每次迭代中，分别计算VNIFGSM和SIM的梯度，
        然后按权重组合，最后更新扰动。
        """
        batch_size = images.shape[0]

        # 初始化动量和当前对抗样本
        momentum = torch.zeros_like(images).to(self.device)
        adv_images = images.clone().detach()

        for step in range(self.epoch):
            adv_images.requires_grad = True

            # 1. 计算VNIFGSM梯度（带方差调优）
            vnifgsm_grad = self._compute_vnifgsm_gradient(adv_images, labels, momentum)

            # 2. 计算SIM梯度（多尺度）
            sim_grad = self._compute_sim_gradient(adv_images, labels)

            # 3. 组合梯度
            combined_grad = (self.momentum_weight * vnifgsm_grad +
                             self.sim_weight * sim_grad)

            # 4. 更新动量（基于组合梯度）
            momentum = self.decay * momentum + combined_grad / torch.mean(
                torch.abs(combined_grad), dim=(1, 2, 3), keepdim=True
            )

            # 5. 更新对抗样本
            adv_images = adv_images.detach() + self.alpha * torch.sign(momentum)

            # 6. 投影到epsilon球和[0,1]范围
            delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

            if (step + 1) % max(1, self.epoch // 5) == 0:
                logging.debug(f"Hybrid attack step {step + 1}/{self.epoch} completed")

        return adv_images


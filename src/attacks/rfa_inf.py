# src/attacks/rfa_inf.py

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_attack import BaseAttack


class RFAInf(BaseAttack):
    """
    Reverse Feature Alignment attack under the Linf threat model (RFA∞).

    This implementation adapts the surrogate-based routine from
    TransferAttackEval's evaluation code, where gradients are taken
    with respect to a robust surrogate network instead of the victim
    model. The attack itself is an iterative Linf-PGD loop driven by
    the surrogate.
    """

    def __init__(
        self,
        model: nn.Module,
        surrogate_model: nn.Module,
        eps: float = 8 / 255,
        alpha: Optional[float] = None,
        steps: int = 10,
        random_start: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ):
        """
        :param model: Victim model wrapped with NormalizedModel (not used for gradients).
        :param surrogate_model: Robust surrogate model that produces gradients.
                                Must accept inputs in [0, 1] and be on the correct device.
        :param eps: Linf budget.
        :param alpha: Step size. Defaults to eps / steps if None.
        :param steps: Number of PGD iterations.
        :param random_start: Whether to start from a random point within the Linf ball.
        :param clamp_min: Minimum valid pixel value.
        :param clamp_max: Maximum valid pixel value.
        """
        super().__init__(model)

        if surrogate_model is None:
            raise ValueError("RFAInf requires a non-None surrogate_model.")

        self.surrogate = surrogate_model.eval()
        for param in self.surrogate.parameters():
            param.requires_grad_(False)

        self.eps = eps
        self.steps = max(1, steps)
        self.alpha = alpha if alpha is not None else eps / self.steps
        self.random_start = random_start
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        logging.info(
            "Initialized RFAInf attack with eps=%.5f, alpha=%.5f, steps=%d, random_start=%s",
            self.eps,
            self.alpha,
            self.steps,
            self.random_start,
        )

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach()
        labels = labels.clone().detach()

        device = images.device
        delta = torch.zeros_like(images, device=device)

        if self.random_start:
            delta.uniform_(-self.eps, self.eps)
            delta = torch.clamp(images + delta, self.clamp_min, self.clamp_max) - images

        for _ in range(self.steps):
            delta.requires_grad_()
            adv = torch.clamp(images + delta, self.clamp_min, self.clamp_max)

            logits = self.surrogate(adv)
            loss = F.cross_entropy(logits, labels)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

            delta = delta + self.alpha * grad.sign()
            delta = torch.clamp(delta, -self.eps, self.eps)
            delta = torch.clamp(images + delta, self.clamp_min, self.clamp_max) - images
            delta = delta.detach()

        adv_images = torch.clamp(images + delta, self.clamp_min, self.clamp_max)
        return adv_images.detach()


# Copyright 2020 Deepmind Technologies Limited.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# A copy of the license is available at http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted for this project to load the Cui2023Decoupled WRN-28-10 checkpoint.

from typing import Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class _Block(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()

        # manual padding to mimic TensorFlow SAME padding
        self.conv_0 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=0,
            bias=False,
        )

        self.batchnorm_1 = nn.BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_1 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError("Unsupported stride.")
        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_planes: int,
        out_planes: int,
        stride: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        blocks = []
        for idx in range(num_blocks):
            blocks.append(
                _Block(
                    in_planes if idx == 0 else out_planes,
                    out_planes,
                    stride if idx == 0 else 1,
                    activation_fn=activation_fn,
                )
            )
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DMWideResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        width: int = 10,
        activation_fn: Type[nn.Module] = nn.ReLU,
        mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
        std: Union[Tuple[float, ...], float] = CIFAR10_STD,
        padding: int = 0,
        num_input_channels: int = 3,
    ):
        super().__init__()
        self.register_buffer(
            "mean",
            torch.tensor(mean).view(num_input_channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(std).view(num_input_channels, 1, 1),
            persistent=False,
        )
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.init_conv = nn.Conv2d(
            num_input_channels,
            num_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer = nn.Sequential(
            _BlockGroup(
                num_blocks,
                num_channels[0],
                num_channels[1],
                1,
                activation_fn=activation_fn,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[1],
                num_channels[2],
                2,
                activation_fn=activation_fn,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[2],
                num_channels[3],
                2,
                activation_fn=activation_fn,
            ),
        )
        self.batchnorm = nn.BatchNorm2d(num_channels[3])
        self.relu = activation_fn()
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


def WideResNet28_10(
    num_classes: int = 10,
    activation_fn: Type[nn.Module] = nn.SiLU,
    mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
    std: Union[Tuple[float, ...], float] = CIFAR10_STD,
    width: int = 10,
):
    return DMWideResNet(
        num_classes=num_classes,
        depth=28,
        width=width,
        activation_fn=activation_fn,
        mean=mean,
        std=std,
    )


def WideResNet94_16(
    num_classes: int = 10,
    activation_fn: Type[nn.Module] = nn.SiLU,
    mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
    std: Union[Tuple[float, ...], float] = CIFAR10_STD,
    width: int = 16,
) -> DMWideResNet:
    return DMWideResNet(
        num_classes=num_classes,
        depth=94,
        width=width,
        activation_fn=activation_fn,
        mean=mean,
        std=std,
    )


def WideResNet70_16(
    num_classes: int = 10,
    activation_fn: Type[nn.Module] = nn.SiLU,
    mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
    std: Union[Tuple[float, ...], float] = CIFAR10_STD,
    width: int = 16,
) -> DMWideResNet:
    return DMWideResNet(
        num_classes=num_classes,
        depth=70,
        width=width,
        activation_fn=activation_fn,
        mean=mean,
        std=std,
    )


__all__ = ["DMWideResNet", "WideResNet28_10", "WideResNet94_16", "WideResNet70_16"]


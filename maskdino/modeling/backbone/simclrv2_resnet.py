# Copyright (c) IDEA, Inc. and its affiliates.
"""SimCLRv2 ResNet encoder, vendored so the SSL backbone needs no extra dependency.

The architecture (ResNet-D stem, bottleneck blocks with selective kernels, and
average-pool shortcut projections) matches the ``lightly-ai`` HuggingFace
checkpoints, which were converted with the ``Separius/SimCLRv2-Pytorch`` script:

    - https://huggingface.co/lightly-ai/simclrv2-imagenet1k-r50_1x_sk1

Checkpoint layout (``torch.load``)::

    {'resnet': {'net.0.*': stem, 'net.1..4.*': stages, 'fc.*': 1000-class head}}

The 1000-class ``fc`` head is **not** part of the downstream image encoder and is
skipped by the loader in :mod:`resnet_ssl_backbone`. Unlike the reference
single-scale wrapper, this encoder is executed stage by stage so all four
residual stages can be exported for detection / instance segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BATCH_NORM_EPSILON",
    "BATCH_NORM_DECAY",
    "BatchNormRelu",
    "PooledBatchNorm2d",
    "DescriptorNorm",
    "conv",
    "SelectiveKernel",
    "Projection",
    "BottleneckBlock",
    "Blocks",
    "Stem",
    "ResNet",
    "SimCLRv2Backbone",
    "get_resnet_layers",
    "adapt_padding_for_stride",
]

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9  # == pytorch's default value as well

#: Stage depths of the supported SimCLRv2 ResNet variants.
_RESNET_LAYERS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
}


def get_resnet_layers(depth: int):
    if depth not in _RESNET_LAYERS:
        raise ValueError(
            f"Unsupported SimCLRv2 ResNet depth {depth}; expected one of {sorted(_RESNET_LAYERS)}."
        )
    return list(_RESNET_LAYERS[depth])


class BatchNormRelu(nn.Sequential):
    def __init__(self, num_channels, relu=True):
        super().__init__(
            nn.BatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON),
            nn.ReLU() if relu else nn.Identity(),
        )


class PooledBatchNorm2d(nn.BatchNorm2d):
    """``BatchNorm2d`` that survives a single value per channel.

    The selective-kernel mixer feeds batch norm a *globally pooled* descriptor of
    shape ``(B, C, 1, 1)``. Estimating batch statistics there requires more than
    one value per channel, so ``F.batch_norm`` raises ``Expected more than 1
    value per channel`` whenever the descriptor holds a single image -- batch
    size 1 on one GPU, the tail of a distributed batch, or a smoke test. The
    descriptor is a global statistic rather than a spatial map, so falling back
    to the running statistics (exactly what eval mode uses) is the only
    meaningful behaviour, and it keeps the layer weight-compatible with the
    pretrained checkpoint.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.numel() // max(input.shape[1], 1) <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(input)


class DescriptorNorm(nn.Sequential):
    """``BatchNormRelu`` for globally pooled ``1x1`` descriptors (SK mixing path)."""

    def __init__(self, num_channels):
        super().__init__(PooledBatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON), nn.ReLU())


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=bias,
    )


def adapt_padding_for_stride(height: int, width: int, stride: int) -> tuple[int, int]:
    """Padding that keeps ``ceil(H/stride)`` correct -- kept for interface parity."""
    return (stride - height % stride) % stride, (stride - width % stride) % stride


class SelectiveKernel(nn.Module):
    """SK conv (SimCLRv2, ``sk_ratio > 0``).

    Splits the activation into two streams and mixes them with softmax weights
    computed from globally pooled activations.
    """

    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(
            conv(in_channels, 2 * out_channels, stride=stride),
            BatchNormRelu(2 * out_channels),
        )
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(
            conv(out_channels, mid_dim, kernel_size=1),
            DescriptorNorm(mid_dim),
            conv(mid_dim, 2 * out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)  # 2, B, C, H, W
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)  # 2, B, C, 1, 1
        return (x * F.softmax(m, dim=0)).sum(dim=0)


class Projection(nn.Module):
    """Shortcut projection for the first block of each stage."""

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # kernel_size = 2 => padding = 1
                nn.AvgPool2d(kernel_size=2, stride=stride, padding=0),
                conv(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = BatchNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [
            conv(in_channels, out_channels, kernel_size=1),
            BatchNormRelu(out_channels),
        ]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(BatchNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(BatchNormRelu(out_channels * 4, relu=False))
        self.net = nn.Sequential(*ops)

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))


class Blocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BottleneckBlock(in_channels, out_channels, stride, sk_ratio, True)]
        )
        self.channels_out = out_channels * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(BottleneckBlock(self.channels_out, out_channels, 1, sk_ratio))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Stem(nn.Sequential):
    """ResNet-D style stem used by the ``_sk1`` SimCLRv2 checkpoints.

    With ``sk_ratio > 0`` it is three 3x3 convolutions (no 7x7) followed by a
    max-pool, producing a stride-4 64-channel map for ``width_multiplier=1``.
    """

    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(BatchNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)


class ResNet(nn.Module):
    """``net`` = ``[Stem, Blocks x4]`` plus the unused 1000-class ``fc`` head."""

    def __init__(self, layers, width_multiplier, sk_ratio):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)
        self.fc = nn.Linear(channels_in, 1000)

    def forward(self, x, apply_fc=False):
        h = self.net(x).mean(dim=[2, 3])
        if apply_fc:
            h = self.fc(h)
        return h


class SimCLRv2Backbone(nn.Module):
    """Image encoder used for fine-tuning: ``net`` (stem + four stages), no ``fc``.

    ``channels_out`` is 2048 for ``width_multiplier=1`` (ResNet-50).
    """

    def __init__(self, depth=50, width_multiplier=1, sk_ratio=0.0625):
        super().__init__()
        self.depth = depth
        self.width_multiplier = width_multiplier
        self.sk_ratio = sk_ratio
        resnet = ResNet(get_resnet_layers(depth), width_multiplier, sk_ratio)
        self.net = resnet.net
        self.channels_out = resnet.channels_out

    def forward(self, x):
        """Dense last-stage feature map ``[B, 2048, H/32, W/32]``."""
        return self.net(x)

    def global_features(self, x):
        return self.net(x).mean(dim=[2, 3])

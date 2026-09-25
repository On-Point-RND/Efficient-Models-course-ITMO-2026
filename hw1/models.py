"""The exact FP32 inference network specified in Homework 1."""

from collections import OrderedDict


def build_model():
    import torch.nn as nn

    return nn.Sequential(OrderedDict([
        ("conv7_s2", nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)),
        ("relu1", nn.ReLU(inplace=True)),
        ("maxpool", nn.MaxPool2d(3, stride=2, padding=1)),
        ("conv5", nn.Conv2d(32, 64, 5, padding=2, bias=False)),
        ("relu2", nn.ReLU(inplace=True)),
        ("conv3_s2_a", nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)),
        ("relu3", nn.ReLU(inplace=True)),
        ("conv1_a", nn.Conv2d(128, 256, 1, bias=False)),
        ("relu4", nn.ReLU(inplace=True)),
        ("conv3_s2_b", nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)),
        ("relu5", nn.ReLU(inplace=True)),
        ("conv1_b", nn.Conv2d(256, 512, 1, bias=False)),
        ("relu6", nn.ReLU(inplace=True)),
        ("gap", nn.AdaptiveAvgPool2d(1)),
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(512, 256)),
        ("relu7", nn.ReLU(inplace=True)),
        ("fc2", nn.Linear(256, 100)),
    ]))

"""CIFAR-10 classifiers on images in [-1, 1], the DDPM output range.

`small` (VGG-like, 1.2M) guides: it is called at every step. `resnet18` (11M)
evaluates. GroupNorm as in the UNet. Logits are divided by `temperature`
(1 = raw): post-hoc calibration is only done on the evaluator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from smc.unet import gn


class Tempered(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, x):
        return self.logits(x) / self.temperature


class SmallVGG(Tempered):
    def __init__(self, widths=(64, 128, 256), n_classes=10):
        super().__init__()
        blocks, c_in = [], 3
        for w in widths:
            blocks += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), gn(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), gn(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(c_in, n_classes)

    def logits(self, x):
        return self.head(self.body(x).mean(dim=(2, 3)))


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.n1 = gn(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.n2 = gn(c_out)
        # The block starts as the identity (gamma = 0 on the last norm):
        # Goyal et al. 2017, ~0.5 pt and a more stable start.
        nn.init.zeros_(self.n2.weight)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), gn(c_out))

    def forward(self, x):
        h = F.relu(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class ResNet18(Tempered):
    def __init__(self, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), gn(64), nn.ReLU(inplace=True))
        stages, c_in = [], 64
        for c_out, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            stages += [ResidualBlock(c_in, c_out, stride), ResidualBlock(c_out, c_out, 1)]
            c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(512, n_classes)

    def logits(self, x):
        return self.head(self.stages(self.stem(x)).mean(dim=(2, 3)))


def build(arch):
    return {"small": SmallVGG, "resnet18": ResNet18}[arch]()


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = build(ckpt["config"]["arch"]).to(device)
    m.load_state_dict(ckpt["model"])
    return m.eval()

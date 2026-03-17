'''
this script is for the network of Project 2.

You can change any parts of this code

-------------------------------------------
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelAttention(nn.Module):
    """Pixel-wise attention: produces a per-pixel gate in [0,1]."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, channels // 8)
        self._gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """Return gate only (B,1,H,W) in [0,1]."""
        return self._gate(x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ChannelAttention(nn.Module):
    """Channel-wise attention (SE-style with avg & max pooling). Returns gate (B,C,1,1)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        gate = self._mlp(avg) + self._mlp(mx)  # (B,C,1,1)
        return gate.clamp_(0, 1)


class SpatialAttention(nn.Module):
    """
    Spatial attention using depthwise multi-kernel factorized convs.
    Produces per-location gate (B,C,H,W) in [0,1].
    """
    def __init__(self, channels: int):
        super().__init__()
        # depthwise convs (groups=channels)
        self.d5_5   = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.d1_7   = nn.Conv2d(channels, channels, (1,7), padding=(0,3), groups=channels, bias=False)
        self.d7_1   = nn.Conv2d(channels, channels, (7,1), padding=(3,0), groups=channels, bias=False)
        self.d1_11  = nn.Conv2d(channels, channels, (1,11), padding=(0,5), groups=channels, bias=False)
        self.d11_1  = nn.Conv2d(channels, channels, (11,1), padding=(5,0), groups=channels, bias=False)
        self.d1_21  = nn.Conv2d(channels, channels, (1,21), padding=(0,10), groups=channels, bias=False)
        self.d21_1  = nn.Conv2d(channels, channels, (21,1), padding=(10,0), groups=channels, bias=False)
        self.fuse   = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.d5_5(x)
        x1 = self.d7_1(self.d1_7(x0))
        x2 = self.d11_1(self.d1_11(x0))
        x3 = self.d21_1(self.d1_21(x0))
        s  = self.fuse(x0 + x1 + x2 + x3)
        return torch.sigmoid(s)  # gate (B,C,H,W)


class SCSPA(nn.Module):
    """
    SCSPA: Synergistic Channel-Spatial-Pixel Attention.
    Outputs a reweighted feature map with residual gating.
    """
    def __init__(self, channels: int, reduction: int = 4, residual: bool = True):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(channels)
        self.pa = PixelAttention(channels)
        self.residual = residual
        # Residual mixing coefficient: start close to identity (0 -> pure out_ca)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca_gate = self.ca(x)           # (B,C,1,1)
        out_ca  = x * ca_gate          # (B,C,H,W)
        sa_gate = self.sa(out_ca)      # (B,C,H,W)
        pa_gate = self.pa.gate(out_ca) # (B,1,H,W)
        out = out_ca * sa_gate * pa_gate
        if self.residual:
            # Residual interpolation between out_ca and out
            out = out_ca + self.gamma * (out - out_ca)
        return out

# ---------------------------
# ResNet-18 with SCSPA hooks
# ---------------------------

def _act():
    return nn.ReLU(inplace=True)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=nn.BatchNorm2d, use_scspa: bool = False, scspa_reduction: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = norm_layer(planes)
        self.act   = _act()
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = norm_layer(planes)
        self.downsample = downsample
        self.scspa = SCSPA(planes, reduction=scspa_reduction) if use_scspa else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.scspa(out)  # <-- attention here (after BN2, before add)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.act(out + identity)
        return out


class Network(nn.Module):
    """
    ResNet-18 (ReLU) with optional SCSPA per stage.
    scspa_stages: tuple of 4 booleans for layer1..layer4.
    scspa_reduction: reduction ratio inside SCSPA's ChannelAttention.
    """
    def __init__(self, num_classes: int = 24,
                 norm_layer=nn.BatchNorm2d,
                 scspa_stages=(False, False, True, True),
                 scspa_reduction: int = 4):
        super().__init__()
        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm_layer(64)
        self.act   = _act()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages [2,2,2,2]
        self.layer1 = self._make_layer(64,  2, stride=1, norm_layer=norm_layer,
                                       use_scspa=scspa_stages[0], scspa_reduction=scspa_reduction)
        self.layer2 = self._make_layer(128, 2, stride=2, norm_layer=norm_layer,
                                       use_scspa=scspa_stages[1], scspa_reduction=scspa_reduction)
        self.layer3 = self._make_layer(256, 2, stride=2, norm_layer=norm_layer,
                                       use_scspa=scspa_stages[2], scspa_reduction=scspa_reduction)
        self.layer4 = self._make_layer(512, 2, stride=2, norm_layer=norm_layer,
                                       use_scspa=scspa_stages[3], scspa_reduction=scspa_reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, planes, blocks, stride, norm_layer, use_scspa, scspa_reduction):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample,
                             norm_layer, use_scspa=use_scspa, scspa_reduction=scspa_reduction)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1, None,
                                     norm_layer, use_scspa=use_scspa, scspa_reduction=scspa_reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        # Zero-init last BN in each BasicBlock to start near identity
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

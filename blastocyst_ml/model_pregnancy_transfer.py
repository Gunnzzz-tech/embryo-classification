"""
Transfer-learning pregnancy-outcome (fertile/non-fertile) model: pretrained
ResNet18 or EfficientNet-B0 image backbone, fused with the 36 clinical
tabular features (hormone panel, blood work, semen analysis), -> binary
pregnant/non-pregnant classification.

Same pretrained-weight source as model_gardner_transfer.py (pytorchcv, via
GitHub release assets) for the same reason: this environment's network
policy blocks torchvision's own weight host (download.pytorch.org).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as pcv_get_model

from model_gardner_transfer import IMAGENET_MEAN, IMAGENET_STD  # noqa: F401 (re-exported)

BACKBONES = ("resnet18", "efficientnet_b0")
_GRADCAM_CHILD = {
    "resnet18": "stage4",
    "efficientnet_b0": "final_block",
}


class PregnancyTransferModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_tabular: int,
        dropout: float = 0.3,
        pretrained: bool = True,
        img_hidden: int = 128,
        tab_hidden: int = 64,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; expected one of {BACKBONES}")
        self.backbone_name = backbone
        self.num_tabular = num_tabular

        net = pcv_get_model(backbone, pretrained=pretrained)
        feat_dim = net.output.in_features if backbone == "resnet18" else net.output.fc.in_features
        self.features = net.features
        self._gradcam_child = _GRADCAM_CHILD[backbone]

        self.dropout = nn.Dropout(dropout)
        self.img_fc = nn.Linear(feat_dim, img_hidden)

        if num_tabular > 0:
            self.tab_fc = nn.Sequential(nn.Linear(num_tabular, tab_hidden), nn.ReLU(), nn.Dropout(dropout))
            fused_dim = img_hidden + tab_hidden
        else:
            self.tab_fc = None
            fused_dim = img_hidden

        self.classifier = nn.Linear(fused_dim, 2)

    def gradcam_target_module(self) -> nn.Module:
        return getattr(self.features, self._gradcam_child)

    def backbone_parameters(self):
        return self.features.parameters()

    def head_parameters(self):
        params = list(self.img_fc.parameters()) + list(self.classifier.parameters())
        if self.tab_fc is not None:
            params += list(self.tab_fc.parameters())
        return params

    def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None):
        feat = self.features(image)
        feat = torch.flatten(feat, 1)
        feat = self.dropout(torch.relu(self.img_fc(feat)))
        if self.tab_fc is not None and tabular is not None and tabular.numel() > 0:
            t = self.tab_fc(tabular)
            feat = torch.cat([feat, t], dim=1)
        return self.classifier(feat)

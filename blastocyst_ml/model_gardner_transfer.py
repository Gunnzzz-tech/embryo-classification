"""
Transfer-learning Gardner grading model: pretrained ResNet18 or EfficientNet-B0
backbone -> shared trunk -> three heads (EXP / ICM / TE), fine-tuned on the
Gardner_train_silver.csv dataset.

Why transfer learning instead of the old from-scratch CNN (model_gardner.py):
a 4-conv-layer CNN trained only on ~2000 embryo microscope images has never
learned general "what does a photograph look like" features, so it is prone
to being confidently wrong on any input unlike its narrow training set (e.g.
a random internet photo). Starting from ImageNet-pretrained weights gives the
network real visual features first, and fine-tuning adapts them -- this
generalizes noticeably better even with a modest dataset like this one.

Pretrained weights come from `pytorchcv` (ImageNet-pretrained, downloaded from
GitHub release assets) rather than torchvision's own weights (hosted on
download.pytorch.org), because this environment's network policy blocks
download.pytorch.org. pytorchcv's resnet18/efficientnet_b0 are independently
trained ImageNet checkpoints with published accuracy close to the standard
torchvision ones -- perfectly fine as a transfer-learning starting point.

Preprocessing note: unlike model_gardner.py (raw [0,1], no normalization),
this model expects standard ImageNet normalization. `image_mean`/`image_std`
are saved into the checkpoint so serving code always uses the preprocessing
that matches whichever checkpoint is loaded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as pcv_get_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONES = ("resnet18", "efficientnet_b0")

# Name of the child module inside `features` whose output is the last spatial
# conv feature map (before the final global-average-pool) -- this is what
# Grad-CAM should hook.
_GRADCAM_CHILD = {
    "resnet18": "stage4",
    "efficientnet_b0": "final_block",
}


class GardnerTransferModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_exp: int,
        num_icm: int,
        num_te: int,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; expected one of {BACKBONES}")
        self.backbone_name = backbone

        # pretrained=False at serving time: we're about to load_state_dict our own
        # fine-tuned weights anyway, so skip the ImageNet download entirely.
        net = pcv_get_model(backbone, pretrained=pretrained)
        if backbone == "resnet18":
            feat_dim = net.output.in_features
        else:  # efficientnet_b0: net.output is Sequential(dropout, fc)
            feat_dim = net.output.fc.in_features

        self.features = net.features  # Sequential ending in global-avg-pool -> (N, C, 1, 1)
        self._gradcam_child = _GRADCAM_CHILD[backbone]

        self.dropout = nn.Dropout(dropout)
        self.head_exp = nn.Linear(feat_dim, num_exp)
        self.head_icm = nn.Linear(feat_dim, num_icm)
        self.head_te = nn.Linear(feat_dim, num_te)

    def gradcam_target_module(self) -> nn.Module:
        """The last spatial conv block (pre-global-pool) to hook Grad-CAM on."""
        return getattr(self.features, self._gradcam_child)

    def backbone_parameters(self):
        return self.features.parameters()

    def head_parameters(self):
        for m in (self.head_exp, self.head_icm, self.head_te):
            yield from m.parameters()

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        feat = self.dropout(feat)
        return self.head_exp(feat), self.head_icm(feat), self.head_te(feat)

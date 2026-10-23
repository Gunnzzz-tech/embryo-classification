"""
CNN for Gardner grading (Expansion / ICM / TE) from a blastocyst image.
Same trunk shape as EmbryoLBModel, with three classification heads instead of one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbryoGardnerModel(nn.Module):
    """Image (3 x H x W) -> (EXP logits, ICM logits, TE logits)."""

    def __init__(
        self,
        num_exp_classes: int = 5,
        num_icm_classes: int = 4,
        num_te_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_exp_classes = num_exp_classes
        self.num_icm_classes = num_icm_classes
        self.num_te_classes = num_te_classes

        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_img = nn.Linear(256, 64)
        self.dropout = nn.Dropout(dropout)

        self.fc_exp = nn.Linear(64, num_exp_classes)
        self.fc_icm = nn.Linear(64, num_icm_classes)
        self.fc_te = nn.Linear(64, num_te_classes)

    def forward(self, image: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(image)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc_img(x)))
        return self.fc_exp(x), self.fc_icm(x), self.fc_te(x)

"""
Small CNN + optional tabular branch for embryo live birth (LB) prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbryoLBModel(nn.Module):
    """Image (3 x H x W) + optional tabular -> binary LB logits."""

    def __init__(
        self,
        image_size: int = 224,
        num_tabular: int = 0,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_tabular = num_tabular
        # CNN: 3 -> 32 -> 64 -> 128 -> 256
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # After pool: 256
        self.fc_img = nn.Linear(256, 64)
        if num_tabular > 0:
            self.fc_tab = nn.Linear(num_tabular, 32)
            self.fc_out = nn.Linear(64 + 32, num_classes)
        else:
            self.fc_tab = None
            self.fc_out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor = None):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc_img(x)))
        if self.num_tabular > 0 and tabular is not None and tabular.numel() > 0:
            t = self.dropout(F.relu(self.fc_tab(tabular)))
            x = torch.cat([x, t], dim=1)
        return self.fc_out(x)

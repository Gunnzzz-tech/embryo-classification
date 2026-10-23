"""
Dataset loader for Gardner_train_silver.csv / Gardner_test_gold_onlyGardnerScores.csv,
shared by train_gardner_transfer.py and eval_gardner_transfer.py.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model_gardner_transfer import IMAGENET_MEAN, IMAGENET_STD

SILVER_COLS = {"exp": "EXP_silver", "icm": "ICM_silver", "te": "TE_silver"}
GOLD_COLS = {"exp": "EXP_gold", "icm": "ICM_gold", "te": "TE_gold"}


def load_gardner_csv(csv_path: str, col_map: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df["Image"] = df["Image"].astype(str).str.strip()
    for label, col in col_map.items():
        if col not in df.columns:
            raise ValueError(f"{csv_path}: expected column {col}")
        # "NA"/"ND" (not assessed) become NaN -> encoded as -1 (ignored in loss/accuracy)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class GardnerDataset(Dataset):
    """Embryo image -> (EXP, ICM, TE) grades. Missing grades are labeled -1 (ignored)."""

    def __init__(
        self,
        image_dir: str,
        df: pd.DataFrame,
        col_map: dict,
        image_size: int = 224,
        train: bool = False,
    ):
        self.image_dir = image_dir
        self.col_map = col_map

        def exists(name: str) -> bool:
            return os.path.isfile(os.path.join(image_dir, name))

        df = df[df["Image"].map(exists)].copy()
        self.df = df.reset_index(drop=True)
        self.transform = build_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, row["Image"])
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)

        labels = {}
        for head, col in self.col_map.items():
            v = row[col]
            labels[head] = int(v) if pd.notna(v) else -1
        return (
            img_t,
            torch.tensor(labels["exp"], dtype=torch.long),
            torch.tensor(labels["icm"], dtype=torch.long),
            torch.tensor(labels["te"], dtype=torch.long),
        )


def class_weights(df: pd.DataFrame, col: str, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights (normalized to mean 1) for CrossEntropyLoss."""
    counts = df[col].value_counts()
    freqs = torch.ones(num_classes)
    for c in range(num_classes):
        freqs[c] = float(counts.get(c, 0)) + 1.0  # +1 smoothing for zero-count classes
    weights = freqs.sum() / (num_classes * freqs)
    return weights

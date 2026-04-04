"""Cleavage embryo dataset from cleaned manifest CSV (image-only classification)."""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CleavageManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label_high_fragmentation"])
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = F.resize(img, [self.image_size, self.image_size])
        return img, torch.tensor(label, dtype=torch.long)


def load_manifest(manifest_path: str):
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    return train_df, val_df

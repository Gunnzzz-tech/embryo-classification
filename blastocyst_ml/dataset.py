"""
Dataset: load embryo images + CSV features for live birth (LB) prediction.
Expects Blastocyst_Dataset with Images/ and Clincial_annotations.csv.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# Default path: same parent as mini_project (e.g. Downloads/Blastocyst_Dataset)
DATA_ROOT = os.environ.get(
    "BLASTOCYST_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "Blastocyst_Dataset"),
)


def _detect_sep(csv_path: str) -> str:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    return ";" if ";" in first else ","


def load_annotations(csv_path: str):
    """Load clinical CSV (semicolon or comma). HVWC-style uses Image,Class -> LB."""
    sep = _detect_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep, encoding="utf-8", on_bad_lines="skip")
    if "Class" in df.columns and "LB" not in df.columns:
        df = df.rename(columns={"Class": "LB"})
    if "Image" not in df.columns and "image" in df.columns:
        df = df.rename(columns={"image": "Image"})
    # Keep key columns present in file
    cols = ["Image", "Fond", "d", "EXP_silver", "ICM_silver", "TE_silver", "Age", "LB"]
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    for c in ["EXP_silver", "ICM_silver", "TE_silver", "d", "LB"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df = df.dropna(subset=["LB"])
    df = df[df["LB"].isin([0, 1])].copy()
    df["LB"] = df["LB"].astype(int)

    return df


class EmbryoLiveBirthDataset(Dataset):
    """Embryo image + tabular features -> Live Birth (0/1)."""

    def __init__(
        self,
        root_dir: str,
        df: pd.DataFrame,
        image_dir: str = "Images",
        image_size: int = 224,
        transform=None,
        use_tabular: bool = True,
    ):
        self.root = root_dir
        img_dir = os.path.join(root_dir, image_dir)
        # Keep only rows whose image file exists
        def exists(im_name):
            p = os.path.join(img_dir, im_name)
            return os.path.isfile(p)
        df = df[df["Image"].map(exists)].copy()
        self.df = df.reset_index(drop=True)
        self.image_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        self.use_tabular = use_tabular
        desired = ["EXP_silver", "ICM_silver", "TE_silver", "d", "Age"]
        available = [c for c in desired if c in self.df.columns]
        if use_tabular and len(available) > 0:
            self.tab_cols = available
            if "Age" in self.tab_cols:
                self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
            self.df[self.tab_cols] = self.df[self.tab_cols].fillna(self.df[self.tab_cols].median())
            self.tab_mean = self.df[self.tab_cols].mean().astype(np.float32)
            self.tab_std = self.df[self.tab_cols].std().replace(0, 1).astype(np.float32)
        else:
            if use_tabular and len(available) == 0:
                import warnings

                warnings.warn(
                    "use_tabular=True but no tabular columns in CSV; using image-only.",
                    UserWarning,
                    stacklevel=2,
                )
            self.tab_cols = []
            self.use_tabular = False
            self.tab_mean = np.zeros(0, dtype=np.float32)
            self.tab_std = np.ones(0, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im_name = row["Image"]
        im_path = os.path.join(self.image_dir, im_name)
        lb = int(row["LB"])

        # Image
        try:
            img = Image.open(im_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))

        img = np.array(img)
        if self.transform:
            img = self.transform(img)
        else:
            import torchvision.transforms.functional as F
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            # F.resize expects (C,H,W) or (N,C,H,W)
            img = F.resize(img, [self.image_size, self.image_size])

        # Tabular (keep float32 for model)
        if self.use_tabular:
            tab = (row[self.tab_cols].values.astype(np.float32) - self.tab_mean.values) / self.tab_std.values
            tab = torch.from_numpy(tab.astype(np.float32))
        else:
            tab = torch.zeros(0)

        return img, tab, torch.tensor(lb, dtype=torch.long)


def get_train_val_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Stratified split by LB when possible."""
    from sklearn.model_selection import train_test_split

    idx = df.index.tolist()
    try:
        idx_train, idx_val = train_test_split(
            idx, test_size=val_frac, stratify=df["LB"], random_state=seed
        )
    except ValueError:
        idx_train, idx_val = train_test_split(idx, test_size=val_frac, random_state=seed)
    return df.loc[idx_train], df.loc[idx_val]

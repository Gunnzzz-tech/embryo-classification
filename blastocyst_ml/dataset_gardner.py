"""
Dataset: load embryo images + Gardner grade CSV (EXP/ICM/TE) for grading.

Expects two CSVs (as shipped by the "Blastocyst Dataset" Gardner-grading release):
  - a "silver" CSV for training, columns: Image;EXP_silver;ICM_silver;TE_silver
  - a "gold" CSV for held-out evaluation, columns: Image;EXP_gold;ICM_gold;TE_gold
Some gold rows have "NA"/"ND" for ICM/TE (not assessed) - those are treated as
missing labels (encoded as -1) and excluded from loss/accuracy for that head only.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

DATA_ROOT = os.environ.get(
    "GARDNER_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "archive-2"),
)

# Maps a logical head name to its column name in the silver/gold CSVs.
SILVER_COLS = {"exp": "EXP_silver", "icm": "ICM_silver", "te": "TE_silver"}
GOLD_COLS = {"exp": "EXP_gold", "icm": "ICM_gold", "te": "TE_gold"}


def load_gardner_csv(csv_path: str, col_map: dict) -> pd.DataFrame:
    """Load a Gardner CSV (semicolon-separated, may have a BOM and a trailing empty column)."""
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df["Image"] = df["Image"].astype(str).str.strip()

    for label, col in col_map.items():
        if col not in df.columns:
            raise ValueError(f"{csv_path}: expected column {col}")
        # "NA"/"ND" (not assessed / not determined) become NaN -> encoded as missing (-1) later
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


class GardnerDataset(Dataset):
    """Blastocyst image -> (EXP, ICM, TE) grades. Missing grades are labeled -1 (ignored in loss)."""

    def __init__(
        self,
        root_dir: str,
        df: pd.DataFrame,
        col_map: dict,
        image_dir: str = "Images/Images",
        image_size: int = 224,
    ):
        self.image_dir = os.path.join(root_dir, image_dir)
        self.image_size = image_size
        self.col_map = col_map

        def exists(name: str) -> bool:
            return os.path.isfile(os.path.join(self.image_dir, name))

        self.df = df[df["Image"].map(exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im_path = os.path.join(self.image_dir, row["Image"])

        try:
            img = Image.open(im_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))

        import torchvision.transforms.functional as TF

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = TF.resize(img, [self.image_size, self.image_size])

        def label_for(head: str) -> int:
            v = row[self.col_map[head]]
            return int(v) if pd.notna(v) else -1

        labels = torch.tensor(
            [label_for("exp"), label_for("icm"), label_for("te")], dtype=torch.long
        )
        return img, labels


def get_train_val_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    """Stratified split by EXP_silver when possible (it's present on almost every row)."""
    from sklearn.model_selection import train_test_split

    idx = df.index.tolist()
    strat_col = SILVER_COLS["exp"] if SILVER_COLS["exp"] in df.columns else None
    try:
        idx_train, idx_val = train_test_split(
            idx, test_size=val_frac, stratify=df[strat_col] if strat_col else None, random_state=seed
        )
    except ValueError:
        idx_train, idx_val = train_test_split(idx, test_size=val_frac, random_state=seed)
    return df.loc[idx_train], df.loc[idx_val]

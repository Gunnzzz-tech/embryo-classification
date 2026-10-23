"""
Dataset loader for the "Embryo dataset" pregnancy-outcome data:
  Data set splitting/{Pregnant,Non-pregnant}/{Cross-validation,External-validation}/<id>.jpg
  labels.xlsx: one row per <id> with `label` (0=non-pregnant, 1=pregnant) and
  36 clinical tabular features (female hormone panel, blood work, male semen
  analysis, etc.)

Cross-validation is what we train/val-split ourselves; External-validation is
the dataset authors' own held-out test set and should only be used for final
evaluation, never for choosing hyperparameters or checkpoints.
"""
from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model_gardner_transfer import IMAGENET_MEAN, IMAGENET_STD

# Chinese column -> short English name. `男方精液处理方法` is already a 0/1 code
# (semen processing method A vs B), so it's used as-is, no one-hot needed.
COLUMN_MAP = {
    "胚胎": "embryo_id",
    "label": "label",
    "女方年龄": "female_age",
    "女方基础FSH": "female_fsh",
    "女方基础PRL": "female_prl",
    "女方基础T": "female_t",
    "女方基础P": "female_p",
    "女方AMH": "female_amh",
    "女方白细胞计数": "female_wbc",
    "女方血小板": "female_platelet",
    "女方血细胞压积": "female_hct",
    "女方凝血常规PT": "female_pt",
    "女方凝血常规APPT": "female_aptt",
    "女方凝血常规TT": "female_tt",
    "女方D二聚体": "female_ddimer",
    "女方甲功TPO": "female_tpo",
    "女方空腹血糖": "female_glucose",
    "女方肾功尿酸": "female_uric_acid",
    "男方浓度": "male_concentration",
    "男方正常形态率": "male_normal_morphology",
    "男方PR": "male_pr",
    "男方NP": "male_np",
    "男方存活率": "male_survival_rate",
    "男方DFI": "male_dfi",
    "男方ALT": "male_alt",
    "男方AST": "male_ast",
    "男方血肌酐": "male_creatinine",
    "男方白细胞": "male_wbc",
    "男方血小板": "male_platelet",
    "男方空腹血糖": "male_glucose",
    "女方用药Gn总天数": "female_gn_days",
    "女方用药HCG日P": "female_hcg_day_p",
    "男方精液处理后NP": "male_np_processed",
    "男方精液处理方法": "male_processing_method",
    "男方精液处理后密度": "male_density_processed",
    "男方精液处理后PR": "male_pr_processed",
    "男方精液处理后IM": "male_im_processed",
    "胚胎冷冻数": "embryo_freeze_count",
}

TAB_COLS = [v for k, v in COLUMN_MAP.items() if v not in ("embryo_id", "label")]


def load_labels(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns=COLUMN_MAP)
    df["embryo_id"] = df["embryo_id"].astype(str).str.strip()
    return df


def list_split(data_root: str, split: str) -> pd.DataFrame:
    """split: 'Cross-validation' or 'External-validation'. Returns id -> folder_label (0/1)."""
    rows = []
    for cls_name, cls_label in (("Pregnant", 1), ("Non-pregnant", 0)):
        d = os.path.join(data_root, "Data set splitting", cls_name, split)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            rows.append({"embryo_id": stem, "folder_label": cls_label, "path": os.path.join(d, fname)})
    return pd.DataFrame(rows)


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


class PregnancyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tab_mean: torch.Tensor,
        tab_std: torch.Tensor,
        image_size: int = 224,
        train: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tab_mean = tab_mean
        self.tab_std = tab_std
        self.transform = build_transform(image_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img_t = self.transform(img)
        tab = torch.tensor([float(row[c]) for c in TAB_COLS], dtype=torch.float32)
        tab = (tab - self.tab_mean) / self.tab_std
        label = int(row["label"])
        return img_t, tab, torch.tensor(label, dtype=torch.long)


def load_merged(data_root: str, xlsx_path: str, split: str) -> pd.DataFrame:
    """Join folder listing (id, folder_label, image path) with xlsx labels/features for one split
    ('Cross-validation' or 'External-validation')."""
    labels = load_labels(xlsx_path)
    listing = list_split(data_root, split)
    merged = listing.merge(labels, on="embryo_id", how="inner")
    mismatched = (merged["folder_label"] != merged["label"]).sum()
    if mismatched:
        print(f"WARNING: {mismatched} rows where folder placement disagrees with xlsx label")
    missing = len(listing) - len(merged)
    if missing:
        print(f"WARNING: {missing}/{len(listing)} images in {split} had no matching xlsx row (dropped)")
    return merged


def build_dataset(data_root: str, xlsx_path: str, split: str, image_size: int, train: bool, tab_mean=None, tab_std=None):
    """Convenience wrapper for a single split with no further train/val split (e.g. final
    External-validation evaluation). Pass tab_mean/tab_std computed from the training split so
    normalization stats never leak from val/test into train. Returns (dataset, tab_mean, tab_std, merged_df)."""
    merged = load_merged(data_root, xlsx_path, split)
    if tab_mean is None or tab_std is None:
        tab_mean = torch.tensor(merged[TAB_COLS].mean().values, dtype=torch.float32)
        tab_std = torch.tensor(merged[TAB_COLS].std().replace(0, 1).values, dtype=torch.float32)

    ds = PregnancyDataset(merged, tab_mean, tab_std, image_size=image_size, train=train)
    return ds, tab_mean, tab_std, merged

#!/usr/bin/env python3
"""
Clean HVWC-style dataset (e.g. hvwc23): train.csv with Image,Class and images under train/.

- Keeps rows with Class in {0, 1}, existing files, optional blur drop
- Writes hvwc23_train_cleaned.csv (semicolon-separated) with Image;LB for train_live_birth.py

Usage:
  python3 clean_hvwc23.py --data ~/Downloads/hvwc23
  python3 train_live_birth.py --data ~/Downloads/hvwc23 --csv hvwc23_train_cleaned.csv \\
      --image-dir train --no-tabular --resume embryo_lb_model.pt
"""
import os
import argparse
import re
import pandas as pd

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

DAY_PREFIX = re.compile(r"^D(\d+)_", re.IGNORECASE)


def blur_score(image_path: str) -> float:
    if not HAS_CV2:
        return 999.0
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def day_from_filename(name: str):
    m = DAY_PREFIX.match(str(name).strip())
    if m:
        return int(m.group(1))
    return pd.NA


def main():
    p = argparse.ArgumentParser(description="Clean HVWC23-style train split (Image, Class)")
    p.add_argument("--data", required=True, help="Dataset root (contains train.csv and train/)")
    p.add_argument("--csv-in", default="train.csv", help="Training labels CSV (comma-separated)")
    p.add_argument("--out", default="hvwc23_train_cleaned.csv", help="Output CSV name (semicolon) in data root")
    p.add_argument("--image-subdir", default="train", help="Subfolder with images")
    p.add_argument("--blur-threshold", type=float, default=0.0, help="Laplacian variance min (0=off)")
    p.add_argument("--skip-blur", action="store_true")
    args = p.parse_args()

    root = os.path.expanduser(args.data)
    csv_in = os.path.join(root, args.csv_in)
    img_dir = os.path.join(root, args.image_subdir)
    out_path = os.path.join(root, args.out)

    if not os.path.isdir(root):
        raise SystemExit(f"Data root not found: {root}")
    if not os.path.isfile(csv_in):
        raise SystemExit(f"CSV not found: {csv_in}")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"Image folder not found: {img_dir}")

    df = pd.read_csv(csv_in, sep=",", encoding="utf-8", on_bad_lines="skip")
    if "Image" not in df.columns or "Class" not in df.columns:
        raise SystemExit(f"Expected columns Image, Class; got {list(df.columns)}")

    n0 = len(df)
    df = df.dropna(subset=["Image", "Class"]).copy()
    df["Image"] = df["Image"].astype(str).str.strip()
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df[df["Class"].isin([0.0, 1.0])].copy()
    df["LB"] = df["Class"].astype(int)
    df = df.drop_duplicates(subset=["Image"], keep="first")

    def img_file_ok(name: str) -> bool:
        return os.path.isfile(os.path.join(img_dir, name))

    before = len(df)
    df = df[df["Image"].map(img_file_ok)].copy()
    print(f"Rows after valid class + dedupe: {before}; missing files removed: {before - len(df)}")

    if HAS_CV2 and not args.skip_blur and args.blur_threshold > 0:
        drop_idx = []
        for idx, row in df.iterrows():
            pth = os.path.join(img_dir, row["Image"])
            if blur_score(pth) < args.blur_threshold:
                drop_idx.append(idx)
        df = df.drop(index=drop_idx).copy()
        print(f"Blur drop: {len(drop_idx)} rows")

    # Developmental day from filename (D3_, D5_) when present
    df["d"] = df["Image"].map(day_from_filename)
    df["d"] = pd.to_numeric(df["d"], errors="coerce")

    out_cols = ["Image", "LB", "d"]
    df[out_cols].to_csv(out_path, sep=";", index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {out_path}")
    print(f"  LB=0: {(df['LB'] == 0).sum()}, LB=1: {(df['LB'] == 1).sum()}")
    if df["d"].notna().any():
        print(f"  d (from filename) value counts:\n{df['d'].value_counts(dropna=True).head(10)}")


if __name__ == "__main__":
    main()

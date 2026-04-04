#!/usr/bin/env python3
"""
Clean Blastocyst_Dataset Clincial_annotations.csv before training.
- Valid LB (0/1), valid Gardner scores (EXP 0-5, ICM 0-3, TE 0-3), valid day (4 or 5)
- Reasonable Age (18-55), drop duplicates, keep only rows with existing image file
- Drop blurry images and their CSV rows (Laplacian variance blur detection)
Output: Clincial_annotations_cleaned.csv in the same data root.
"""
import os
import argparse
import pandas as pd

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

DATA_ROOT = os.environ.get(
    "BLASTOCYST_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "Blastocyst_Dataset"),
)


def blur_score(image_path: str) -> float:
    """
    Laplacian variance: higher = sharper, lower = blurrier.
    Returns 0.0 on read error (treated as drop).
    """
    if not HAS_CV2:
        return 999.0  # skip dropping
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clean(df: pd.DataFrame, image_dir: str, blur_threshold: float = 0, skip_blur: bool = False, blur_dropped_list: list = None) -> tuple:
    """Apply cleaning rules; return cleaned DataFrame. If blur_dropped_list is provided, append dropped image names for blur."""
    report = []
    if blur_dropped_list is None:
        blur_dropped_list = []
    n_start = len(df)

    # 1. Drop rows with missing Image
    df = df[df["Image"].notna() & (df["Image"].astype(str).str.strip() != "")].copy()
    report.append(("Missing or empty Image", n_start - len(df)))
    n_start = len(df)

    # 2. LB must be 0 or 1
    if "LB" in df.columns:
        df["LB"] = pd.to_numeric(df["LB"], errors="coerce")
        before = len(df)
        df = df[df["LB"].isin([0.0, 1.0])].copy()
        df["LB"] = df["LB"].astype(int)
        report.append(("Invalid or missing LB (not 0/1)", before - len(df)))
        n_start = len(df)

    # 3. Gardner scores: EXP 0-5, ICM 0-3, TE 0-3
    for col, low, high in [
        ("EXP_silver", 0, 5),
        ("ICM_silver", 0, 3),
        ("TE_silver", 0, 3),
    ]:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        before = len(df)
        df = df[df[col].between(low, high, inclusive="both")].copy()
        report.append((f"{col} out of range [{low},{high}]", before - len(df)))
        n_start = len(df)

    # 4. Day d in {4, 5}
    if "d" in df.columns:
        df["d"] = pd.to_numeric(df["d"], errors="coerce")
        before = len(df)
        df = df[df["d"].isin([4, 5])].copy()
        report.append(("Invalid day d (not 4 or 5)", before - len(df)))
        n_start = len(df)

    # 5. Age: 18-55 if present; else leave as-is and fill NaN with median later
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        before = len(df)
        # Keep only reasonable age; drop rows with Age outside 18-55 (optional: or cap/fill)
        df = df[df["Age"].isna() | df["Age"].between(18, 55, inclusive="both")].copy()
        report.append(("Age out of range [18,55] or invalid", before - len(df)))

    # 6. Duplicate Image: keep first
    before = len(df)
    df = df.drop_duplicates(subset=["Image"], keep="first").copy()
    report.append(("Duplicate Image", before - len(df)))

    # 7. Only rows where image file exists
    def image_exists(name):
        p = os.path.join(image_dir, name.strip())
        return os.path.isfile(p)

    before = len(df)
    df = df[df["Image"].astype(str).map(image_exists)].copy()
    report.append(("Image file not found", before - len(df)))

    # 8. Drop blurry images and their CSV rows (keep CSV in sync with image set)
    if HAS_CV2 and not skip_blur and blur_threshold > 0:
        before = len(df)
        drop_indices = []
        for idx, row in df.iterrows():
            im_name = str(row["Image"]).strip()
            im_path = os.path.join(image_dir, im_name)
            score = blur_score(im_path)
            if score < blur_threshold:
                drop_indices.append(idx)
                blur_dropped_list.append(im_name)
        df = df.drop(index=drop_indices).copy()
        report.append(("Blurry image (Laplacian variance < threshold)", before - len(df)))

    return df, report


def main():
    p = argparse.ArgumentParser(description="Clean Blastocyst annotations CSV")
    p.add_argument("--data", default=DATA_ROOT, help="Path to Blastocyst_Dataset root")
    p.add_argument("--csv", default="Clincial_annotations.csv", help="Input CSV name")
    p.add_argument("--out", default="Clincial_annotations_cleaned.csv", help="Output CSV name")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding (try latin-1 if utf-8 fails)")
    p.add_argument("--blur-threshold", type=float, default=50.0, help="Drop images with Laplacian variance below this (0 = disable). Typical: 50–200.")
    p.add_argument("--skip-blur", action="store_true", help="Do not check for blurry images (e.g. if opencv not installed)")
    p.add_argument("--blur-dropped", default="", help="Optional: save list of dropped (blurry) image filenames to this file")
    args = p.parse_args()

    data_root = args.data
    csv_path = os.path.join(data_root, args.csv)
    image_dir = os.path.join(data_root, "Images")
    out_path = os.path.join(data_root, args.out)

    if not os.path.isdir(data_root):
        raise SystemExit(f"Data root not found: {data_root}")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    if not os.path.isdir(image_dir):
        raise SystemExit(f"Images dir not found: {image_dir}")

    # Load with fallback encoding
    try:
        df = pd.read_csv(csv_path, sep=";", encoding=args.encoding, on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=";", encoding="latin-1", on_bad_lines="skip")

    if not HAS_CV2 and args.blur_threshold > 0 and not args.skip_blur:
        print("Warning: opencv not installed; skipping blur check. Install opencv-python-headless or use --skip-blur.")

    blur_dropped_list = []
    n_initial = len(df)
    df, report = clean(
        df, image_dir,
        blur_threshold=args.blur_threshold,
        skip_blur=args.skip_blur,
        blur_dropped_list=blur_dropped_list,
    )
    n_final = len(df)

    if args.blur_dropped and blur_dropped_list:
        dropped_path = os.path.join(data_root, args.blur_dropped) if not os.path.isabs(args.blur_dropped) else args.blur_dropped
        with open(dropped_path, "w", encoding="utf-8") as f:
            f.write("\n".join(blur_dropped_list))
        print(f"Dropped blurry image list saved: {dropped_path} ({len(blur_dropped_list)} files)")

    # Ensure columns we need exist
    required = ["Image", "EXP_silver", "ICM_silver", "TE_silver", "d", "LB"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"After cleaning, missing column: {c}")

    df.to_csv(out_path, sep=";", index=False, encoding="utf-8")
    print("Cleaning report:")
    for reason, count in report:
        if count > 0:
            print(f"  - {reason}: {count} rows removed")
    print(f"  Initial rows: {n_initial}")
    print(f"  Final rows:   {n_final}")
    if "LB" in df.columns:
        print(f"  LB=0: {(df['LB']==0).sum()}, LB=1: {(df['LB']==1).sum()}")
    print(f"Saved: {out_path}")
    print("Use this file for training: --csv Clincial_annotations_cleaned.csv")


if __name__ == "__main__":
    main()

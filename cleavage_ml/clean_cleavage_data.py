#!/usr/bin/env python3
"""
Clean CleavageEmbryov1.1 (COCO format): verify images exist, optional blur filter,
derive labels from annotations (cell vs fragment counts).

Output: cleavage_manifest_cleaned.csv in dataset root with:
  split, file_name, image_path, num_cells, num_fragments, fragment_ratio,
  label_high_fragmentation (1 if fragment_ratio >= train median after cleaning)
"""
import os
import json
import argparse
import pandas as pd

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

DEFAULT_ROOT = os.path.expanduser("~/Downloads/CleavageEmbryov1.1")


def laplacian_var(path: str) -> float:
    if not HAS_CV2:
        return 999.0
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_split(data_root: str, split: str):
    ann_path = os.path.join(data_root, "annotations", f"instances_{split}.json")
    img_root = os.path.join(data_root, "images", split)
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    cell_n = {}
    frag_n = {}
    for a in coco["annotations"]:
        iid = a["image_id"]
        if a["category_id"] == 1:
            cell_n[iid] = cell_n.get(iid, 0) + 1
        else:
            frag_n[iid] = frag_n.get(iid, 0) + 1

    rows = []
    for im in coco["images"]:
        iid = im["id"]
        fn = im["file_name"]
        path = os.path.join(img_root, fn)
        nc = cell_n.get(iid, 0)
        nf = frag_n.get(iid, 0)
        tot = nc + nf
        ratio = (nf / tot) if tot > 0 else 0.0
        rows.append(
            {
                "split": split,
                "file_name": fn,
                "image_path": path,
                "num_cells": nc,
                "num_fragments": nf,
                "fragment_ratio": ratio,
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description="Clean CleavageEmbryo COCO dataset")
    p.add_argument("--data", default=DEFAULT_ROOT, help="CleavageEmbryov1.1 root")
    p.add_argument("--out", default="cleavage_manifest_cleaned.csv", help="Output CSV name (in data root)")
    p.add_argument("--blur-threshold", type=float, default=0.0, help="Drop if Laplacian var below this (0=off)")
    p.add_argument("--skip-blur", action="store_true")
    p.add_argument("--blur-dropped", default="", help="Write list of dropped filenames")
    args = p.parse_args()

    root = os.path.expanduser(args.data)
    if not os.path.isdir(root):
        raise SystemExit(f"Dataset root not found: {root}")

    rows = load_split(root, "train") + load_split(root, "val")
    df = pd.DataFrame(rows)

    report = []
    n0 = len(df)

    # Missing file
    df["_exists"] = df["image_path"].map(os.path.isfile)
    report.append(("Image file missing", int((~df["_exists"]).sum())))
    df = df[df["_exists"]].drop(columns=["_exists"])

    # No annotations at all (0 cells and 0 fragments) — drop
    before = len(df)
    df = df[(df["num_cells"] + df["num_fragments"]) > 0].copy()
    report.append(("No cell/fragment annotations", before - len(df)))

    # Blur
    dropped_blur = []
    if HAS_CV2 and not args.skip_blur and args.blur_threshold > 0:
        before = len(df)
        mask = []
        for _, r in df.iterrows():
            v = laplacian_var(r["image_path"])
            ok = v >= args.blur_threshold
            mask.append(ok)
            if not ok:
                dropped_blur.append(r["file_name"])
        keep_idx = [i for i, m in enumerate(mask) if m]
        df = df.iloc[keep_idx].reset_index(drop=True)
        report.append(("Blurry image", before - len(df)))

    # Label: high fragmentation if fragment_ratio >= median of TRAIN (after clean)
    train_df = df[df["split"] == "train"]
    if len(train_df) == 0:
        raise SystemExit("No train rows left after cleaning.")
    med = train_df["fragment_ratio"].median()
    df["label_high_fragmentation"] = (df["fragment_ratio"] >= med).astype(int)

    out_path = os.path.join(root, args.out)
    df_out = df[
        [
            "split",
            "file_name",
            "image_path",
            "num_cells",
            "num_fragments",
            "fragment_ratio",
            "label_high_fragmentation",
        ]
    ]
    df_out.to_csv(out_path, index=False)

    if args.blur_dropped and dropped_blur:
        bp = args.blur_dropped if os.path.isabs(args.blur_dropped) else os.path.join(root, args.blur_dropped)
        with open(bp, "w", encoding="utf-8") as f:
            f.write("\n".join(dropped_blur))
        print(f"Blur dropped list: {bp}")

    print("Cleaning report:")
    for name, cnt in report:
        if cnt:
            print(f"  - {name}: {cnt}")
    print(f"  Initial rows: {n0}")
    print(f"  Final rows:   {len(df)}")
    print(f"  Train median fragment_ratio (for label): {med:.4f}")
    print(f"  Label=1 count: {(df['label_high_fragmentation']==1).sum()}, Label=0: {(df['label_high_fragmentation']==0).sum()}")
    print(f"Saved: {out_path}")
    print("Train with: python3 train_cleavage.py --manifest path/to/cleavage_manifest_cleaned.csv")


if __name__ == "__main__":
    main()

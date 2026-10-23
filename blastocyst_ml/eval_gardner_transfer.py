#!/usr/bin/env python3
"""
Evaluate one or more gardner_transfer checkpoints against the held-out gold
test set (Gardner_test_gold_onlyGardnerScores.csv). Some gold rows have
NA/ND for ICM/TE (not assessed) -- those are excluded from that head's
accuracy only (encoded as -1 by dataset_transfer.load_gardner_csv).

Usage:
  python3 eval_gardner_transfer.py --data-dir . gardner_resnet18.pt gardner_efficientnet_b0.pt
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset_transfer import GOLD_COLS, GardnerDataset, load_gardner_csv
from model_gardner_transfer import GardnerTransferModel
from train_gardner_transfer import evaluate


def load_checkpoint_model(path: str) -> tuple[GardnerTransferModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GardnerTransferModel(
        ckpt["backbone"], ckpt["num_exp"], ckpt["num_icm"], ckpt["num_te"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--csv", default="Gardner_test_gold_onlyGardnerScores.csv")
    ap.add_argument("--image-dir", default="Images/Images")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()

    csv_path = os.path.join(args.data_dir, args.csv)
    image_dir = os.path.join(args.data_dir, args.image_dir)
    df = load_gardner_csv(csv_path, GOLD_COLS)
    print(f"Gold rows with an existing image: checking {len(df)} candidates...")

    for ckpt_path in args.checkpoints:
        model, ckpt = load_checkpoint_model(ckpt_path)
        image_size = ckpt.get("image_size", 224)
        ds = GardnerDataset(image_dir, df, GOLD_COLS, image_size=image_size, train=False)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        acc = evaluate(model, loader, torch.device("cpu"))
        print(
            f"\n=== {ckpt_path} (backbone={ckpt['backbone']}, n={len(ds)}, "
            f"train val_avg was {ckpt.get('val_acc', {}).get('avg', float('nan')):.3f}) ==="
        )
        print(
            f"  gold exp_acc={acc['exp']:.3f}  icm_acc={acc['icm']:.3f}  "
            f"te_acc={acc['te']:.3f}  avg={acc['avg']:.3f}"
        )


if __name__ == "__main__":
    main()

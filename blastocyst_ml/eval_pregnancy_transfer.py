#!/usr/bin/env python3
"""
Evaluate one or more pregnancy_transfer checkpoints against the dataset's own
held-out External-validation set (never touched during training).

Usage:
  python3 eval_pregnancy_transfer.py pregnancy_resnet18.pt pregnancy_efficientnet_b0.pt
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset_pregnancy import PregnancyDataset, load_merged
from model_pregnancy_transfer import PregnancyTransferModel
from train_pregnancy_transfer import evaluate


def load_checkpoint_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PregnancyTransferModel(ckpt["backbone"], num_tabular=ckpt["num_tabular"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--xlsx", default="labels.xlsx")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()

    merged = load_merged(args.data_root, args.xlsx, "External-validation")
    print(f"External-validation: {len(merged)} images, label counts {merged['label'].value_counts().to_dict()}")

    for ckpt_path in args.checkpoints:
        model, ckpt = load_checkpoint_model(ckpt_path)
        tab_mean = torch.tensor(ckpt["tab_mean"], dtype=torch.float32)
        tab_std = torch.tensor(ckpt["tab_std"], dtype=torch.float32)
        ds = PregnancyDataset(merged, tab_mean, tab_std, image_size=ckpt.get("image_size", 224), train=False)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        m = evaluate(model, loader, torch.device("cpu"))
        print(
            f"\n=== {ckpt_path} (backbone={ckpt['backbone']}, cross-val best f1 was "
            f"{ckpt.get('val_metrics', {}).get('f1', float('nan')):.3f}) ==="
        )
        print(
            f"  external acc={m['acc']:.3f}  precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  n={m['n']}"
        )


if __name__ == "__main__":
    main()

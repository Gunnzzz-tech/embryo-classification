#!/usr/bin/env python3
"""
Fine-tune a pretrained backbone (ResNet18 or EfficientNet-B0) + the 36
clinical tabular features on the Cross-validation split of the Embryo
dataset, to predict Pregnant (1) vs Non-pregnant (0).

Trains/validates only on Cross-validation (our own stratified split);
External-validation is reserved for eval_pregnancy_transfer.py's final,
unbiased check.

Usage:
  python3 train_pregnancy_transfer.py --backbone resnet18 --epochs 12 --save pregnancy_resnet18.pt
  python3 train_pregnancy_transfer.py --backbone efficientnet_b0 --epochs 12 --save pregnancy_efficientnet_b0.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_pregnancy import TAB_COLS, PregnancyDataset, load_merged
from model_gardner_transfer import IMAGENET_MEAN, IMAGENET_STD
from model_pregnancy_transfer import PregnancyTransferModel


def stratified_split(df, val_frac: float, seed: int):
    train_parts, val_parts = [], []
    for label, group in df.groupby("label"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = int(len(group) * val_frac)
        val_parts.append(group.iloc[:n_val])
        train_parts.append(group.iloc[n_val:])
    import pandas as pd

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct, total = 0, 0
    tp = fp = tn = fn = 0
    for im, tab, y in loader:
        im, tab, y = im.to(device), tab.to(device), y.to(device)
        logits = model(im, tab)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())
    acc = correct / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "n": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".", help="Folder with 'Data set splitting/'")
    ap.add_argument("--xlsx", default="labels.xlsx")
    ap.add_argument("--backbone", required=True, choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    merged = load_merged(args.data_root, args.xlsx, "Cross-validation")
    train_df, val_df = stratified_split(merged, args.val_frac, args.seed)
    print(f"train:{len(train_df)} val:{len(val_df)}  train label counts: {train_df['label'].value_counts().to_dict()}", flush=True)

    tab_mean = torch.tensor(train_df[TAB_COLS].mean().values, dtype=torch.float32)
    tab_std = torch.tensor(train_df[TAB_COLS].std().replace(0, 1).values, dtype=torch.float32)

    train_ds = PregnancyDataset(train_df, tab_mean, tab_std, image_size=args.size, train=True)
    val_ds = PregnancyDataset(val_df, tab_mean, tab_std, image_size=args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    n0 = (train_df["label"] == 0).sum()
    n1 = (train_df["label"] == 1).sum()
    class_w = torch.tensor([len(train_df) / (2 * n0), len(train_df) / (2 * n1)], dtype=torch.float32).to(device)
    print(f"Class weights (non-pregnant, pregnant): {class_w.tolist()}", flush=True)

    model = PregnancyTransferModel(args.backbone, num_tabular=len(TAB_COLS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": args.lr * args.backbone_lr_mult},
            {"params": model.head_parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss, n_batches = 0.0, 0
        for im, tab, y in train_loader:
            im, tab, y = im.to(device), tab.to(device), y.to(device)
            opt.zero_grad()
            logits = model(im, tab)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss /= max(n_batches, 1)

        m = evaluate(model, val_loader, device)
        dt = time.time() - t0
        is_best = m["f1"] > best_f1
        if is_best:
            best_f1 = m["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": args.backbone,
                    "num_tabular": len(TAB_COLS),
                    "tab_cols": TAB_COLS,
                    "tab_mean": tab_mean.tolist(),
                    "tab_std": tab_std.tolist(),
                    "image_size": args.size,
                    "image_mean": IMAGENET_MEAN,
                    "image_std": IMAGENET_STD,
                    "val_metrics": m,
                },
                args.save,
            )
        print(
            f"Epoch {epoch+1}/{args.epochs} ({dt:.0f}s) train_loss={train_loss:.4f} "
            f"val_acc={m['acc']:.3f} val_precision={m['precision']:.3f} val_recall={m['recall']:.3f} "
            f"val_f1={m['f1']:.3f} best_f1={best_f1:.3f}{'  <-- saved' if is_best else ''}",
            flush=True,
        )
    print(f"Done. Best val F1: {best_f1:.4f}. Saved to {args.save}", flush=True)


if __name__ == "__main__":
    main()

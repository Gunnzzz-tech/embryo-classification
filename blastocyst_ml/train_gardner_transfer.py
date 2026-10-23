#!/usr/bin/env python3
"""
Fine-tune a pretrained backbone (ResNet18 or EfficientNet-B0) on
Gardner_train_silver.csv for EXP/ICM/TE grading.

Usage:
  python3 train_gardner_transfer.py --backbone resnet18 --epochs 15 --save gardner_resnet18.pt
  python3 train_gardner_transfer.py --backbone efficientnet_b0 --epochs 15 --save gardner_efficientnet_b0.pt
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_transfer import SILVER_COLS, GardnerDataset, class_weights, load_gardner_csv
from model_gardner_transfer import IMAGENET_MEAN, IMAGENET_STD, GardnerTransferModel


def split_df(df: pd.DataFrame, val_frac: float, seed: int):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_frac)
    return df.iloc[n_val:].reset_index(drop=True), df.iloc[:n_val].reset_index(drop=True)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = {"exp": 0, "icm": 0, "te": 0}
    total = {"exp": 0, "icm": 0, "te": 0}
    for im, y_exp, y_icm, y_te in loader:
        im = im.to(device)
        logits_exp, logits_icm, logits_te = model(im)
        for name, logits, y in (("exp", logits_exp, y_exp), ("icm", logits_icm, y_icm), ("te", logits_te, y_te)):
            y = y.to(device)
            mask = y >= 0
            if mask.sum() == 0:
                continue
            pred = logits.argmax(dim=1)
            correct[name] += int((pred[mask] == y[mask]).sum().item())
            total[name] += int(mask.sum().item())
    acc = {name: (correct[name] / total[name] if total[name] else float("nan")) for name in correct}
    acc["avg"] = float(np.nanmean([acc["exp"], acc["icm"], acc["te"]]))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Folder with Images/Images and the CSVs")
    ap.add_argument("--csv", default="Gardner_train_silver.csv")
    ap.add_argument("--image-dir", default="Images/Images")
    ap.add_argument("--backbone", required=True, choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Debug: cap dataset size")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    csv_path = os.path.join(args.data_dir, args.csv)
    image_dir = os.path.join(args.data_dir, args.image_dir)
    df = load_gardner_csv(csv_path, SILVER_COLS)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    df_train, df_val = split_df(df, args.val_frac, args.seed)

    num_exp = int(df[SILVER_COLS["exp"]].max()) + 1
    num_icm = int(df[SILVER_COLS["icm"]].max()) + 1
    num_te = int(df[SILVER_COLS["te"]].max()) + 1
    print(f"Classes -> exp:{num_exp} icm:{num_icm} te:{num_te}  train:{len(df_train)} val:{len(df_val)}", flush=True)

    w_exp = class_weights(df_train, SILVER_COLS["exp"], num_exp).to(device)
    w_icm = class_weights(df_train, SILVER_COLS["icm"], num_icm).to(device)
    w_te = class_weights(df_train, SILVER_COLS["te"], num_te).to(device)
    print(f"Class weights exp={w_exp.tolist()}", flush=True)
    print(f"Class weights icm={w_icm.tolist()}", flush=True)
    print(f"Class weights te={w_te.tolist()}", flush=True)

    train_ds = GardnerDataset(image_dir, df_train, SILVER_COLS, image_size=args.size, train=True)
    val_ds = GardnerDataset(image_dir, df_val, SILVER_COLS, image_size=args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = GardnerTransferModel(args.backbone, num_exp, num_icm, num_te).to(device)
    crit_exp = nn.CrossEntropyLoss(weight=w_exp, ignore_index=-1)
    crit_icm = nn.CrossEntropyLoss(weight=w_icm, ignore_index=-1)
    crit_te = nn.CrossEntropyLoss(weight=w_te, ignore_index=-1)

    opt = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": args.lr * args.backbone_lr_mult},
            {"params": model.head_parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_avg = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for im, y_exp, y_icm, y_te in train_loader:
            im, y_exp, y_icm, y_te = im.to(device), y_exp.to(device), y_icm.to(device), y_te.to(device)
            opt.zero_grad()
            logits_exp, logits_icm, logits_te = model(im)
            loss = crit_exp(logits_exp, y_exp) + crit_icm(logits_icm, y_icm) + crit_te(logits_te, y_te)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss /= max(n_batches, 1)

        acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        is_best = acc["avg"] > best_avg
        if is_best:
            best_avg = acc["avg"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": args.backbone,
                    "num_exp": num_exp,
                    "num_icm": num_icm,
                    "num_te": num_te,
                    "image_size": args.size,
                    "image_mean": IMAGENET_MEAN,
                    "image_std": IMAGENET_STD,
                    "val_acc": acc,
                },
                args.save,
            )
        print(
            f"Epoch {epoch+1}/{args.epochs} ({dt:.0f}s) train_loss={train_loss:.4f} "
            f"val_exp={acc['exp']:.3f} val_icm={acc['icm']:.3f} val_te={acc['te']:.3f} "
            f"val_avg={acc['avg']:.3f} best_avg={best_avg:.3f}{'  <-- saved' if is_best else ''}",
            flush=True,
        )
    print(f"Done. Best val avg accuracy: {best_avg:.4f}. Saved to {args.save}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a model to predict Gardner grades (Expansion, ICM, TE) from blastocyst images.

Uses the Gardner-grading release of the Blastocyst Dataset:
  - Gardner_train_silver.csv   (silver-standard labels, used for train/val)
  - Gardner_test_gold_onlyGardnerScores.csv   (gold-standard labels, held out, evaluated once at the end)
  - Images/Images/*.png

Usage:
  export GARDNER_DATA=/path/to/archive-2   # optional, defaults to ../../archive-2
  pip install -r requirements.txt
  python3 train_gardner.py
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_gardner import (
    DATA_ROOT,
    GOLD_COLS,
    SILVER_COLS,
    GardnerDataset,
    get_train_val_split,
    load_gardner_csv,
)
from model_gardner import EmbryoGardnerModel

HEADS = ["exp", "icm", "te"]


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """Accuracy over rows where labels != -1 (missing/not-assessed). None if no valid rows."""
    mask = labels != -1
    if mask.sum().item() == 0:
        return None
    pred = logits.argmax(dim=1)
    return (pred[mask] == labels[mask]).float().mean().item()


def run_eval(model, loader, device):
    model.eval()
    correct = {h: 0 for h in HEADS}
    total = {h: 0 for h in HEADS}
    with torch.no_grad():
        for img, labels in loader:
            img, labels = img.to(device), labels.to(device)
            logits = model(img)
            for i, h in enumerate(HEADS):
                mask = labels[:, i] != -1
                if mask.sum().item() == 0:
                    continue
                pred = logits[i].argmax(dim=1)
                correct[h] += (pred[mask] == labels[mask, i]).sum().item()
                total[h] += mask.sum().item()
    return {h: (correct[h] / total[h] if total[h] else None) for h in HEADS}


def main():
    p = argparse.ArgumentParser(description="Train Gardner grade predictor (EXP/ICM/TE) from images")
    p.add_argument("--data", default=DATA_ROOT, help="Path to dataset root (e.g. ~/Downloads/archive-2)")
    p.add_argument("--train-csv", default="Gardner_train_silver.csv")
    p.add_argument("--test-csv", default="Gardner_test_gold_onlyGardnerScores.csv")
    p.add_argument("--image-dir", default="Images/Images", help="Subfolder of --data with image files")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="embryo_gardner_model.pt", help="Output model path")
    args = p.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Data dir not found: {args.data}")
    train_csv_path = os.path.join(args.data, args.train_csv)
    test_csv_path = os.path.join(args.data, args.test_csv)
    if not os.path.isfile(train_csv_path):
        raise SystemExit(f"Train CSV not found: {train_csv_path}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    silver_df = load_gardner_csv(train_csv_path, SILVER_COLS)
    df_train, df_val = get_train_val_split(silver_df, val_frac=args.val_frac, seed=args.seed)

    num_exp = int(silver_df[SILVER_COLS["exp"]].max()) + 1
    num_icm = int(silver_df[SILVER_COLS["icm"]].max()) + 1
    num_te = int(silver_df[SILVER_COLS["te"]].max()) + 1
    print(f"Classes -> EXP: {num_exp}, ICM: {num_icm}, TE: {num_te}", flush=True)

    train_ds = GardnerDataset(args.data, df_train, SILVER_COLS, image_dir=args.image_dir, image_size=args.size)
    val_ds = GardnerDataset(args.data, df_val, SILVER_COLS, image_dir=args.image_dir, image_size=args.size)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = EmbryoGardnerModel(num_exp, num_icm, num_te).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for img, labels in train_loader:
            img, labels = img.to(device), labels.to(device)
            opt.zero_grad()
            logits_exp, logits_icm, logits_te = model(img)
            loss = (
                criterion(logits_exp, labels[:, 0])
                + criterion(logits_icm, labels[:, 1])
                + criterion(logits_te, labels[:, 2])
            )
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_acc = run_eval(model, val_loader, device)
        valid_accs = [a for a in val_acc.values() if a is not None]
        score = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": args.size,
                    "num_exp": num_exp,
                    "num_icm": num_icm,
                    "num_te": num_te,
                },
                args.save,
            )
        acc_str = ", ".join(f"{h}={val_acc[h]:.4f}" if val_acc[h] is not None else f"{h}=n/a" for h in HEADS)
        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val[{acc_str}]  avg={score:.4f}  best_avg={best_score:.4f}",
            flush=True,
        )

    print(f"Done. Best val avg accuracy: {best_score:.4f}. Model saved to {args.save}", flush=True)

    if os.path.isfile(test_csv_path):
        gold_df = load_gardner_csv(test_csv_path, GOLD_COLS)
        test_ds = GardnerDataset(args.data, gold_df, GOLD_COLS, image_dir=args.image_dir, image_size=args.size)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

        best = torch.load(args.save, map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        test_acc = run_eval(model, test_loader, device)
        acc_str = ", ".join(f"{h}={test_acc[h]:.4f}" if test_acc[h] is not None else f"{h}=n/a" for h in HEADS)
        print(f"Held-out gold test accuracy -> {acc_str}", flush=True)
    else:
        print(f"No test CSV found at {test_csv_path}; skipping held-out evaluation.", flush=True)


if __name__ == "__main__":
    main()

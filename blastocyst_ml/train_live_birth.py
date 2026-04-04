#!/usr/bin/env python3
"""
Train a model to predict live birth (LB) from embryo images + CSV features.
Uses Blastocyst_Dataset: Images/ + Clincial_annotations.csv.

Usage:
  export BLASTOCYST_DATA=/path/to/Blastocyst_Dataset   # optional
  pip install -r requirements.txt
  python3 train_live_birth.py
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DATA_ROOT, load_annotations, EmbryoLiveBirthDataset, get_train_val_split
from model import EmbryoLBModel


def load_matching_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> tuple[int, int]:
    """Load checkpoint tensors into model when shapes match (partial transfer)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"No 'model' key in {ckpt_path}")
    sd = ckpt["model"]
    cur = model.state_dict()
    n_ok, n_skip = 0, 0
    for k in cur.keys():
        if k not in sd:
            n_skip += 1
            continue
        if cur[k].shape != sd[k].shape:
            n_skip += 1
            continue
        cur[k] = sd[k].clone().to(device)
        n_ok += 1
    model.load_state_dict(cur)
    return n_ok, n_skip


def main():
    p = argparse.ArgumentParser(description="Train embryo LB predictor (image + CSV)")
    p.add_argument("--data", default=DATA_ROOT, help="Path to Blastocyst_Dataset root (e.g. ~/Downloads/Blastocyst_Dataset)")
    p.add_argument("--csv", default="Clincial_annotations_cleaned.csv", help="CSV filename inside data root (use cleaned after clean_data.py)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--no-tabular", action="store_true", help="Use image only")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="embryo_lb_model.pt", help="Output model path")
    p.add_argument(
        "--image-dir",
        default="Images",
        help="Subfolder of --data with image files (e.g. Images for Blastocyst, train for HVWC23)",
    )
    p.add_argument(
        "--resume",
        default="",
        help="Path to .pt checkpoint: load weights where tensor shapes match (fine-tune / transfer)",
    )
    args = p.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Data dir not found: {args.data}")

    csv_path = os.path.join(args.data, args.csv)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = load_annotations(csv_path)
    df_train, df_val = get_train_val_split(df, val_frac=args.val_frac, seed=args.seed)

    train_ds = EmbryoLiveBirthDataset(
        args.data,
        df_train,
        image_dir=args.image_dir,
        image_size=args.size,
        use_tabular=not args.no_tabular,
    )
    val_ds = EmbryoLiveBirthDataset(
        args.data,
        df_val,
        image_dir=args.image_dir,
        image_size=args.size,
        use_tabular=not args.no_tabular,
    )
    num_tab = len(train_ds.tab_cols) if not args.no_tabular else 0
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Tabular dim: {num_tab}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = EmbryoLBModel(image_size=args.size, num_tabular=num_tab, num_classes=2).to(device)
    if args.resume:
        rp = os.path.expanduser(args.resume)
        if not os.path.isfile(rp):
            raise SystemExit(f"--resume file not found: {rp}")
        ok, skip = load_matching_weights(model, rp, device)
        print(f"Resume from {rp}: loaded {ok} tensors, skipped {skip} (shape/name mismatch)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for im, tab, lb in train_loader:
            im, tab, lb = im.to(device), tab.to(device).float(), lb.to(device)
            opt.zero_grad()
            logits = model(im, tab if num_tab else None)
            loss = criterion(logits, lb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for im, tab, lb in val_loader:
                im, tab, lb = im.to(device), tab.to(device).float(), lb.to(device)
                logits = model(im, tab if num_tab else None)
                pred = logits.argmax(dim=1)
                correct += (pred == lb).sum().item()
                total += lb.size(0)
        val_acc = correct / total if total else 0
        if val_acc > best_acc:
            best_acc = val_acc
            payload = {
                "model": model.state_dict(),
                "tab_cols": train_ds.tab_cols if num_tab else [],
                "tab_mean": train_ds.tab_mean.values.tolist() if num_tab else [],
                "tab_std": train_ds.tab_std.values.tolist() if num_tab else [],
                "image_size": args.size,
                "num_tabular": num_tab,
            }
            torch.save(payload, args.save)
        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  best={best_acc:.4f}",
            flush=True,
        )
    print(f"Done. Best val accuracy: {best_acc:.4f}. Model saved to {args.save}", flush=True)


if __name__ == "__main__":
    main()

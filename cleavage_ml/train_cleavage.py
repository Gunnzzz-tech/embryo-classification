#!/usr/bin/env python3
"""
Train image-only classifier: high vs low fragmentation (from CleavageEmbryo manifest).
Run clean_cleavage_data.py first to create cleavage_manifest_cleaned.csv.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse CNN from blastocyst_ml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "blastocyst_ml"))
from model import EmbryoLBModel

from dataset_cleavage import CleavageManifestDataset, load_manifest

DEFAULT_MANIFEST = os.path.expanduser(
    "~/Downloads/CleavageEmbryov1.1/cleavage_manifest_cleaned.csv"
)


def main():
    p = argparse.ArgumentParser(description="Train cleavage fragmentation classifier")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to cleavage_manifest_cleaned.csv")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--save", default="cleavage_fragmentation_model.pt")
    args = p.parse_args()

    if not os.path.isfile(args.manifest):
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Run: python3 clean_cleavage_data.py --data ~/Downloads/CleavageEmbryov1.1"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_df, val_df = load_manifest(args.manifest)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}", flush=True)

    train_ds = CleavageManifestDataset(train_df, image_size=args.size)
    val_ds = CleavageManifestDataset(val_df, image_size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = EmbryoLBModel(image_size=args.size, num_tabular=0, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = args.save
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(__file__), save_path)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for im, lb in train_loader:
            im, lb = im.to(device), lb.to(device)
            opt.zero_grad()
            logits = model(im, None)
            loss = criterion(logits, lb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for im, lb in val_loader:
                im, lb = im.to(device), lb.to(device)
                pred = model(im, None).argmax(dim=1)
                correct += (pred == lb).sum().item()
                total += lb.size(0)
        val_acc = correct / total if total else 0
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": "cleavage_high_fragmentation",
                    "label_meaning": "1 = fragment_ratio >= train_median",
                },
                save_path,
            )
        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  best={best_acc:.4f}",
            flush=True,
        )

    print(f"Done. Best val acc: {best_acc:.4f}. Saved: {save_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate a Grad-CAM heatmap overlay for a single embryo image, for the
pregnancy-outcome (fertile/non-fertile) model.

Usage:
    python3 visualize_heatmap_pregnancy.py path/to/embryo_image.jpg
    python3 visualize_heatmap_pregnancy.py path/to/embryo_image.jpg --ckpt ./pregnancy_resnet18.pt --tabular tabular.json

--tabular (optional): path to a JSON file of {column_name: value} for any subset of the 36
clinical features (see dataset_pregnancy.TAB_COLS for exact names). Omitted columns fall back
to the training-set mean.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from gradcam import overlay_heatmap
from gradcam_pregnancy import PregnancyGradCAM, build_tab_vector, load_model

LABELS = {0: "Non-pregnant", 1: "Pregnant"}


def add_caption(img_rgb: np.ndarray, lines: list[str] | str) -> np.ndarray:
    if isinstance(lines, str):
        lines = [lines]
    h, w = img_rgb.shape[:2]
    font_scale = max(0.32, min(0.5, w / 480.0))
    line_h = int(22 * font_scale / 0.4)
    strip_h = 10 + line_h * len(lines)
    canvas = np.full((h + strip_h, w, 3), 24, dtype=np.uint8)
    canvas[:h, :, :] = img_rgb
    for i, text in enumerate(lines):
        y = h + int(line_h * 0.75) + i * line_h
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM heatmap for the pregnancy-outcome model")
    ap.add_argument("image")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "pregnancy_resnet18.pt"))
    ap.add_argument("--tabular", default="", help="Optional path to a JSON file of clinical feature values")
    ap.add_argument("--out-dir", default="./heatmap_out")
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    model, image_size, preprocess, tab_cols, tab_mean, tab_std = load_model(args.ckpt)
    print(f"Loaded {args.ckpt} (backbone={model.backbone_name})")
    cam_engine = PregnancyGradCAM(model)

    values = {}
    if args.tabular:
        with open(args.tabular, encoding="utf-8") as f:
            values = json.load(f)
        missing = [c for c in tab_cols if c not in values]
        if missing:
            print(f"Note: {len(missing)} clinical fields not in --tabular, using training-set mean: {missing}")
    else:
        print("No --tabular given: all 36 clinical fields default to their training-set mean.")

    tab = build_tab_vector(tab_cols, tab_mean, tab_std, values)

    with open(args.image, "rb") as f:
        raw = f.read()
    image_tensor = preprocess(raw, image_size)

    display = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    display = display - display.min()
    if display.max() > 1e-8:
        display = display / display.max()
    original_rgb = (display * 255).astype(np.uint8)

    cam, predicted_class, probs = cam_engine(image_tensor, tab)
    confidence = float(probs[predicted_class])
    overlay = overlay_heatmap(original_rgb, cam, alpha=args.alpha)

    caption_lines = [LABELS[predicted_class], f"confidence {confidence*100:.1f}%"]
    out_path = os.path.join(args.out_dir, f"{base_name}_pregnancy_heatmap.png")
    Image.fromarray(add_caption(overlay, caption_lines)).save(out_path)

    combined = np.concatenate(
        [add_caption(original_rgb.copy(), ["Original", ""]), add_caption(overlay, caption_lines)], axis=1
    )
    combined_path = os.path.join(args.out_dir, f"{base_name}_pregnancy_combined.png")
    Image.fromarray(combined).save(combined_path)

    cam_engine.remove_hooks()
    print(f"Wrote {out_path}")
    print(f"Wrote {combined_path}")
    print(f"Prediction: {LABELS[predicted_class]}  confidence={confidence:.4f}")
    print(f"Probabilities [non_pregnant, pregnant]: {[round(p, 4) for p in probs.tolist()]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Grad-CAM heatmap overlays for a single embryo image, for all three
Gardner heads (EXP / ICM / TE) predicted by embryo_gardner_model.pt.

Usage:
    python3 visualize_heatmap.py path/to/embryo_image.png
    python3 visualize_heatmap.py path/to/embryo_image.png --ckpt ./embryo_gardner_model.pt --out-dir ./heatmaps

Writes, into --out-dir (default: ./heatmap_out):
    <name>_exp_heatmap.png   <name>_icm_heatmap.png   <name>_te_heatmap.png
    <name>_combined.png      (original + all three overlays side by side, with scores)
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from gradcam import GardnerGradCAM, load_model, overlay_heatmap, preprocess_image

HEAD_LABELS = {
    "exp": "Expansion (EXP)",
    "icm": "Inner Cell Mass (ICM)",
    "te": "Trophectoderm (TE)",
}


def add_caption(img_rgb: np.ndarray, lines: list[str] | str) -> np.ndarray:
    """Adds a dark caption strip under the image with one or more lines of text,
    sized to the image width so text doesn't overflow into the next panel."""
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
        cv2.putText(
            canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM heatmaps for the Gardner grading model")
    ap.add_argument("image", help="Path to an embryo image (png/jpg)")
    ap.add_argument(
        "--ckpt",
        default=os.path.join(os.path.dirname(__file__), "embryo_gardner_model.pt"),
        help="Path to embryo_gardner_model.pt",
    )
    ap.add_argument("--out-dir", default="./heatmap_out")
    ap.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity (0-1)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    model, image_size = load_model(args.ckpt)
    cam_engine = GardnerGradCAM(model)

    with open(args.image, "rb") as f:
        raw = f.read()
    image_tensor = preprocess_image(raw, image_size)  # 1x3xSxS, matches model input exactly

    # Original image resized to the model's input size, for a pixel-aligned overlay.
    original_rgb = (image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    panels = [add_caption(original_rgb.copy(), ["Original", ""])]
    summary_lines = []

    for head in ("exp", "icm", "te"):
        cam, predicted_class, probs = cam_engine(image_tensor, head)
        confidence = float(probs[predicted_class])
        overlay = overlay_heatmap(original_rgb, cam, alpha=args.alpha)

        caption_lines = [HEAD_LABELS[head], f"class {predicted_class}  ({confidence*100:.1f}%)"]
        out_path = os.path.join(args.out_dir, f"{base_name}_{head}_heatmap.png")
        Image.fromarray(add_caption(overlay, caption_lines)).save(out_path)
        print(f"Wrote {out_path}  ->  predicted {head}={predicted_class}, confidence={confidence:.4f}")
        print(f"    all-class probabilities: {[round(p, 4) for p in probs.tolist()]}")

        panels.append(add_caption(overlay, caption_lines))
        summary_lines.append(f"{HEAD_LABELS[head]}: class {predicted_class} ({confidence*100:.1f}%)")

    cam_engine.remove_hooks()

    combined = np.concatenate(panels, axis=1)
    combined_path = os.path.join(args.out_dir, f"{base_name}_combined.png")
    Image.fromarray(combined).save(combined_path)
    print(f"\nWrote combined figure -> {combined_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

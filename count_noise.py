#!/usr/bin/env python3
"""
Count how many images in your train/val splits have empty YOLO label files (pure noise).
"""

import argparse
from pathlib import Path

def count_noise(img_dir: Path, lbl_dir: Path, exts=('*.jpg','*.jpeg','*.png')):
    imgs = []
    for pat in exts:
        imgs.extend(img_dir.rglob(pat))
    total = len(imgs)
    noise = 0
    for img in imgs:
        lbl = lbl_dir / f"{img.stem}.txt"
        # If the .txt exists but is zero bytes => noise
        if lbl.stat().st_size == 0:
            noise += 1
    return total, noise

def main():
    p = argparse.ArgumentParser(description="Count pure-noise images in YOLO splits")
    p.add_argument('--train-images', type=Path, required=True, help="Train images folder")
    p.add_argument('--train-labels', type=Path, required=True, help="Train labels folder")
    p.add_argument('--val-images',   type=Path, required=True, help="Val images folder")
    p.add_argument('--val-labels',   type=Path, required=True, help="Val labels folder")
    args = p.parse_args()

    for split in ('train', 'val'):
        img_dir = getattr(args, f"{split}_images")
        lbl_dir = getattr(args, f"{split}_labels")
        total, noise = count_noise(img_dir, lbl_dir)
        pct = noise / total * 100 if total else 0
        print(f"{split.capitalize()}: {noise}/{total} images are pure noise ({pct:.1f}%)")

if __name__ == "__main__":
    main()

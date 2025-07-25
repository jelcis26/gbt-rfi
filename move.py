import argparse
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
LBL_EXTS = {'.txt', '.xml', '.json'}  # adjust if needed

def gather_files(dirs, exts):
    files = {}
    for d in dirs:
        for path in Path(d).rglob('*'):
            if path.suffix.lower() in exts:
                files[path.stem] = path
    return files

def make_dirs(root):
    for split in ('train', 'val'):
        for typ in ('images', 'labels'):
            (root / split / typ).mkdir(parents=True, exist_ok=True)

def copy_pair(img_path, lbl_path, root, split):
    dst_img = root / split / 'images' / img_path.name
    dst_lbl = root / split / 'labels' / lbl_path.name
    shutil.copy2(img_path, dst_img)
    shutil.copy2(lbl_path, dst_lbl)

def main():
    parser = argparse.ArgumentParser(description="Combine images & labels and split 80/20")
    parser.add_argument('--images',   nargs='+', required=True, help="Paths to image folders")
    parser.add_argument('--labels',   nargs='+', required=True, help="Paths to label folders")
    parser.add_argument('--output',   required=True, help="Output dataset root")
    parser.add_argument('--seed',     type=int, default=42, help="Random seed")
    parser.add_argument('--split',    type=float, default=0.8, help="Train fraction (default 0.8)")
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = Path(args.output)
    make_dirs(out_root)

    # 1) Gather
    imgs = gather_files(args.images, IMG_EXTS)
    lbls = gather_files(args.labels, LBL_EXTS)

    # 2) Match basename keys
    keys = sorted(set(imgs) & set(lbls))
    if not keys:
        print("No matching image/label basenames found.")
        return

    # 3) Shuffle & split
    random.shuffle(keys)
    split_idx = int(len(keys) * args.split)
    train_keys = keys[:split_idx]
    val_keys   = keys[split_idx:]

    # 4) Copy
    for k in train_keys:
        copy_pair(imgs[k], lbls[k], out_root, 'train')
    for k in val_keys:
        copy_pair(imgs[k], lbls[k], out_root, 'val')

    print(f"Total pairs: {len(keys)} → train: {len(train_keys)}, val: {len(val_keys)}")
    print(f"Data written under {out_root}")

if __name__ == '__main__':
    main()

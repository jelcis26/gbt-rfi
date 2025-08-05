import os
import shutil

# Paths — adjust as needed
BASE         = '/datax/scratch/jliang/dataset_forests_merges'
VIS          = '/datax/scratch/jliang/dataset_forests_merges/visualization'
OUT          = '/datax/scratch/jliang/aaai_supplementary/forests'  # where we’ll save the pairs
MAX_SAMPLES  = 10    # ← limit here

# ——— INPUT PATHS ———
TRAIN_LABELS = os.path.join(BASE, 'train', 'labels')
TRAIN_IMAGES = os.path.join(BASE, 'train', 'images')
VIS_TRAIN    = os.path.join(VIS, 'train')

# ——— OUTPUT PATHS ———
OUT_IMAGES = os.path.join(OUT, 'images')
OUT_VIS    = os.path.join(OUT, 'visualization')
OUT_LABELS = os.path.join(OUT, 'labels')

for d in (OUT_IMAGES, OUT_VIS, OUT_LABELS):
    os.makedirs(d, exist_ok=True)

count = 0
for lbl_fname in os.listdir(TRAIN_LABELS):
    if count >= MAX_SAMPLES:
        break

    # only care about .txt label files
    if not lbl_fname.lower().endswith('.txt'):
        continue

    lbl_path = os.path.join(TRAIN_LABELS, lbl_fname)
    # skip empty labels
    if os.path.getsize(lbl_path) == 0:
        continue

    base, _ = os.path.splitext(lbl_fname)

    # find the corresponding train image
    img_src = None
    for ext in ('.jpg', '.jpeg', '.png'):
        candidate = os.path.join(TRAIN_IMAGES, base + ext)
        if os.path.exists(candidate):
            img_src = candidate
            break
    if img_src is None:
        continue

    # find the corresponding visualization
    vis_src = os.path.join(VIS_TRAIN, os.path.basename(img_src))
    if not os.path.exists(vis_src):
        continue

    # copy image, visualization, and label
    shutil.copy2(img_src,   os.path.join(OUT_IMAGES, os.path.basename(img_src)))
    shutil.copy2(vis_src,   os.path.join(OUT_VIS,    os.path.basename(vis_src)))
    shutil.copy2(lbl_path,  os.path.join(OUT_LABELS, lbl_fname))

    count += 1

print(f'Done — sampled {count} examples.')

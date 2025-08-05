import os
import random
import shutil

# adjust this to wherever your dataset folder lives
BASE = '/datax/scratch/jliang/dataset_final_7'

train_img = os.path.join(BASE, 'train', 'images')
train_lbl = os.path.join(BASE, 'train', 'labels')
val_img   = os.path.join(BASE, 'val',   'images')
val_lbl   = os.path.join(BASE, 'val',   'labels')

# collect all image filenames in train/images
imgs = [f for f in os.listdir(train_img) if os.path.isfile(os.path.join(train_img, f))]
random.shuffle(imgs)

# compute how many to move (20%)
n_val = int(len(imgs) * 0.2)

for fname in imgs[:n_val]:
    # src/dst for image
    src_i = os.path.join(train_img, fname)
    dst_i = os.path.join(val_img,   fname)
    shutil.move(src_i, dst_i)

    # matching label: same base name but .txt
    base, _ = os.path.splitext(fname)
    lbl = base + '.txt'
    src_l = os.path.join(train_lbl, lbl)
    dst_l = os.path.join(val_lbl,   lbl)
    if os.path.exists(src_l):
        shutil.move(src_l, dst_l)
    else:
        print(f'⚠️  label missing for {fname}')

print(f'Moved {n_val} images + labels into val/, leaving {len(imgs)-n_val} in train/')

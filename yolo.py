#!/usr/bin/env python3
import random
import numpy as np
import torch
from ultralytics import YOLO

# 1) Deterministic setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# 2) Instantiate model once
model = YOLO("yolov8n.pt")

# 3) Base overrides (including seed)
overrides = {
    "data": "/home/jliang/gbt-rfi/data.yaml",
    "epochs": 100,
    "batch": 32,
    "workers": 2,
    "lr0": 0.001,
    "warmup_epochs": 5.0,
    "cls": 0.3,
    "dfl": 0.3,
    "iou": 0.3,
    "mosaic": False,
    "rect": True,
    "fliplr": 0.0,
    "patience": 20,
    "seed": SEED,            # make YOLO’s internal RNG use the same seed
    "single_cls" : True,  # single class training
    "name": "yolo-final-v8",
}

# 4) OOM‐backoff loop (optional)
while True:
    try:
        model.train(**overrides)
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and overrides["batch"] > 1:
            overrides["batch"] //= 2
            print(f"[OOM] retrying with batch={overrides['batch']}…")
            torch.cuda.empty_cache()
        else:
            raise



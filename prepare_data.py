"""
prepare_data.py
---------------
Step 2: Read the .h5 file in chunks (gzip-compressed, so we never load
        all 2.7 GB at once), resize every image to IMG_SIZE x IMG_SIZE,
        do a stratified 70/15/15 split, compute train mean/std, and save
        three .npz files: data/train.npz, data/val.npz, data/test.npz.

Augmentation is NOT done here — it happens on the GPU in train.py.
Split is done BEFORE computing mean/std so there is no data leakage.

Run:
    D:\Gpu-shit\nasa\Scripts\python.exe prepare_data.py
"""

import h5py
import numpy as np
from PIL import Image           # for resizing (Pillow)
import os

# ── Config (change these for ablation) ──────────────────────────────────────
H5_PATH  = r"D:\New Downloads\Galaxy10_DECals_NoDuplicated.h5"
IMG_SIZE = 96          # resize target (change to 128 for a bigger-input experiment)
SEED     = 42          # fixed seed for reproducibility
OUT_DIR  = "data"      # output folder for .npz files
CHUNK    = 500         # how many images to read + resize at a time (RAM budget)

IMG_KEY   = "images"
LABEL_KEY = "ans"

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load labels only (tiny — a few KB) ───────────────────────────────────
print("Reading labels...")
with h5py.File(H5_PATH, "r") as f:
    labels = f[LABEL_KEY][:]          # (N,) uint8
    N      = f[IMG_KEY].shape[0]

print(f"Total samples: {N}  |  Image size target: {IMG_SIZE}x{IMG_SIZE}")

# ── 2. Stratified split indices ──────────────────────────────────────────────
# We build train/val/test index lists class-by-class so each split has
# the same class proportions as the full dataset (stratified).
train_idx, val_idx, test_idx = [], [], []

for cls in range(10):
    idx = np.where(labels == cls)[0]
    np.random.shuffle(idx)             # shuffle within class
    n  = len(idx)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    # rest goes to test (avoids rounding losing samples)
    train_idx.extend(idx[:n_train])
    val_idx.extend(idx[n_train:n_train + n_val])
    test_idx.extend(idx[n_train + n_val:])

# Sort indices so h5py reads are sequential (faster on compressed files)
train_idx = np.array(sorted(train_idx))
val_idx   = np.array(sorted(val_idx))
test_idx  = np.array(sorted(test_idx))

print(f"Split sizes — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

# ── 3. Helper: read + resize a list of indices from the .h5 file ────────────
def load_and_resize(h5file, indices, img_size, chunk_size=CHUNK):
    """
    Read images at `indices` from an open h5py file, resize each to
    (img_size, img_size), return a uint8 numpy array (N, C, H, W).
    Processes `chunk_size` images at a time to cap RAM usage.
    Channel-first (C, H, W) because PyTorch expects that layout.
    """
    n = len(indices)
    out = np.empty((n, 3, img_size, img_size), dtype=np.uint8)

    for start in range(0, n, chunk_size):
        end        = min(start + chunk_size, n)
        batch_idx  = indices[start:end]

        # h5py fancy indexing on a sorted list is reasonably fast
        raw = h5file[IMG_KEY][batch_idx]   # (chunk, 256, 256, 3) uint8

        for j, img_hw in enumerate(raw):
            # PIL resize: BILINEAR is a good balance of speed vs quality
            pil_img = Image.fromarray(img_hw).resize(
                (img_size, img_size), Image.BILINEAR
            )
            # Convert HWC → CHW for PyTorch
            out[start + j] = np.array(pil_img).transpose(2, 0, 1)

        print(f"  processed {end}/{n}", end="\r")

    print()
    return out

# ── 4. Read and resize each split ───────────────────────────────────────────
print("\nProcessing train split...")
with h5py.File(H5_PATH, "r") as f:
    X_train = load_and_resize(f, train_idx, IMG_SIZE)
y_train = labels[train_idx]

print("Processing val split...")
with h5py.File(H5_PATH, "r") as f:
    X_val = load_and_resize(f, val_idx, IMG_SIZE)
y_val = labels[val_idx]

print("Processing test split...")
with h5py.File(H5_PATH, "r") as f:
    X_test = load_and_resize(f, test_idx, IMG_SIZE)
y_test = labels[test_idx]

# ── 5. Compute mean and std from TRAIN set only ──────────────────────────────
# We compute per-channel mean/std on the TRAIN set so val/test cannot
# leak any statistics into the normalisation.
# X_train is uint8 (0-255) → divide by 255 to get float [0,1] first.
print("\nComputing train mean and std (per channel)...")
X_train_f = X_train.astype(np.float32) / 255.0   # (N, 3, H, W) float32

# axis=(0,2,3) means: average over samples, height, and width → shape (3,)
mean = X_train_f.mean(axis=(0, 2, 3))
std  = X_train_f.std(axis=(0, 2, 3))
print(f"  mean (R,G,B): {mean}")
print(f"  std  (R,G,B): {std}")

# ── 6. Save .npz files ───────────────────────────────────────────────────────
# We save uint8 images (not normalised floats) to keep file sizes small.
# Normalisation is applied on-the-fly in train.py using the saved mean/std.
print("\nSaving .npz files...")

np.savez_compressed(
    os.path.join(OUT_DIR, "train.npz"),
    images=X_train, labels=y_train, mean=mean, std=std
)
np.savez_compressed(
    os.path.join(OUT_DIR, "val.npz"),
    images=X_val, labels=y_val
)
np.savez_compressed(
    os.path.join(OUT_DIR, "test.npz"),
    images=X_test, labels=y_test
)

print(f"Saved:")
print(f"  {OUT_DIR}/train.npz  ({X_train.shape[0]} images)")
print(f"  {OUT_DIR}/val.npz    ({X_val.shape[0]} images)")
print(f"  {OUT_DIR}/test.npz   ({X_test.shape[0]} images)")
print("\nDone. Run train.py next.")

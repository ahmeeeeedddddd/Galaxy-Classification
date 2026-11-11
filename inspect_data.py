"""
inspect_data.py
---------------
Step 1: Open the .h5 file, print everything we need to know about it,
        and save a visual grid so we can actually SEE the images.

Key design decision: we do NOT load all images into RAM here.
We only read labels (tiny) + one image per class (10 images).
This makes the script run in seconds instead of minutes.

Run:
    D:\Gpu-shit\nasa\Scripts\python.exe inspect_data.py
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no GUI backend — write file only, never blocks
import matplotlib.pyplot as plt
import os

# ── Path configuration (single place to change) ─────────────────────────────
H5_PATH  = r"D:\New Downloads\Galaxy10_DECals_NoDuplicated.h5"
OUT_DIR  = "outputs"
GRID_OUT = os.path.join(OUT_DIR, "class_grid.png")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "0 Disturbed",
    "1 Merging",
    "2 Round smooth",
    "3 In-between round smooth",
    "4 Cigar-shaped smooth",
    "5 Barred spiral",
    "6 Unbarred tight spiral",
    "7 Unbarred loose spiral",
    "8 Edge-on without bulge",
    "9 Edge-on with bulge",
]

# ── 1. Open file and print structure ────────────────────────────────────────
print("=" * 60)
print(f"Opening: {H5_PATH}")
print("=" * 60)

with h5py.File(H5_PATH, "r") as f:

    # Discover every dataset key in the file (we never assume key names)
    keys = list(f.keys())
    print(f"\nTop-level keys: {keys}")

    print("\nDataset details:")
    for k in keys:
        ds = f[k]
        # Also check if the dataset is compressed
        chunks   = ds.chunks
        compress = ds.compression
        print(f"  [{k}]  shape={ds.shape}  dtype={ds.dtype}  "
              f"chunks={chunks}  compression={compress}")

    # ── 2. Load LABELS only (tiny — a few KB) ───────────────────────────────
    # We read labels fully because they're tiny and we need them to find
    # the first occurrence of each class without scanning all images.
    img_key   = "images"   # update if key printout above shows a different name
    label_key = "ans"      # update if key printout above shows a different name

    labels     = f[label_key][:]          # (N,) — loads in milliseconds
    n_total    = f[img_key].shape[0]
    img_shape  = f[img_key].shape[1:]     # (H, W, C) — just metadata, no read
    img_dtype  = f[img_key].dtype

    # ── 3. Print statistics ─────────────────────────────────────────────────
    print(f"\nTotal images : {n_total}")
    print(f"Image shape  : {img_shape[0]} x {img_shape[1]} x {img_shape[2]}  (H x W x C)")
    print(f"Image dtype  : {img_dtype}")
    print(f"Label shape  : {labels.shape}  dtype={labels.dtype}")
    print(f"Label range  : min={labels.min()}  max={labels.max()}")

    counts = np.bincount(labels.astype(int), minlength=10)
    print("\nClass distribution:")
    print(f"  {'Class':<35} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*52}")
    for i, (name, cnt) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"  {name:<35} {cnt:>6}  {cnt/n_total*100:>5.1f}%")

    # ── 4. Read ONE image per class (10 reads total, not N) ─────────────────
    # h5py supports integer-index reads: f["images"][i] reads only row i.
    # This is fast even for compressed datasets because HDF5 decompresses
    # only the chunk containing index i.
    print(f"\nReading one example per class (10 single-row reads)...")
    example_imgs = {}
    for cls_idx in range(10):
        matches = np.where(labels.astype(int) == cls_idx)[0]
        if len(matches) == 0:
            print(f"  WARNING: class {cls_idx} has no samples!")
            continue
        first_idx = int(matches[0])
        example_imgs[cls_idx] = f[img_key][first_idx]  # single image read
        print(f"  Class {cls_idx}: found {len(matches):>5} samples, "
              f"using index {first_idx}")

# ── 5. Save the class grid image ────────────────────────────────────────────
print(f"\nBuilding class grid → {GRID_OUT}")

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for cls_idx in range(10):
    if cls_idx not in example_imgs:
        axes[cls_idx].axis("off")
        axes[cls_idx].set_title(f"Class {cls_idx}\n(no samples)", fontsize=9)
        continue
    axes[cls_idx].imshow(example_imgs[cls_idx])
    axes[cls_idx].set_title(CLASS_NAMES[cls_idx], fontsize=9, pad=4)
    axes[cls_idx].axis("off")

plt.suptitle("Galaxy10 DECaLS — one example per class", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(GRID_OUT, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved → {os.path.abspath(GRID_OUT)}")
print("\nDone.")
print("=" * 60)

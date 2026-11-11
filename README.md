# Galaxy Classification — CNN from Scratch

A PyTorch CNN trained entirely from scratch on the **Galaxy10 DECaLS** dataset (17,675 galaxy images, 10 classes). Built for a computer vision college project.

---

## Dataset
**Galaxy10 DECaLS** (no duplicates) — downloaded from [Zenodo](https://zenodo.org/record/5765106).  
Place the file at `D:\New Downloads\Galaxy10_DECals_NoDuplicated.h5` (path is set in `inspect_data.py` and `prepare_data.py`).

| Class | Name | Count |
|---|---|---|
| 0 | Disturbed | 1,081 |
| 1 | Merging | 1,853 |
| 2 | Round smooth | 2,640 |
| 3 | In-between round smooth | 2,021 |
| 4 | Cigar-shaped smooth | 333 |
| 5 | Barred spiral | 2,040 |
| 6 | Unbarred tight spiral | 1,823 |
| 7 | Unbarred loose spiral | 2,600 |
| 8 | Edge-on without bulge | 1,419 |
| 9 | Edge-on with bulge | 1,865 |

---

## File Structure
```
├── inspect_data.py    # Explore the .h5 file, print stats, save class grid
├── prepare_data.py    # Resize + stratified 70/15/15 split → data/*.npz
├── model.py           # CNN architecture (4 conv blocks, GAP, dropout)
├── train.py           # Training loop (AdamW, OneCycleLR, early stopping)
├── evaluate.py        # Test-set evaluation, confusion matrix, misclassified grid
├── requirements.txt
└── outputs/           # Created automatically (curves, confusion matrix, etc.)
```

---

## Setup

```bash
D:\Gpu-shit\nasa\Scripts\pip.exe install -r requirements.txt
```

---

## Run Order

```bash
# 1. Inspect the dataset
D:\Gpu-shit\nasa\Scripts\python.exe inspect_data.py

# 2. Prepare and split the data (takes ~5-10 min — reads + resizes 17k images)
D:\Gpu-shit\nasa\Scripts\python.exe prepare_data.py

# 3. Verify the model architecture and param count
D:\Gpu-shit\nasa\Scripts\python.exe model.py

# 4. Train (saves best_model.pth and curves.png to outputs/)
D:\Gpu-shit\nasa\Scripts\python.exe train.py

# 5. Evaluate on the test set (run ONCE after training)
D:\Gpu-shit\nasa\Scripts\python.exe evaluate.py
```

---

## CNN Architecture

```
Input  (B, 3, 96, 96)
  → Block 1: Conv(3→32)  + BN + ReLU + MaxPool  → (B, 32,  48, 48)
  → Block 2: Conv(32→64) + BN + ReLU + MaxPool  → (B, 64,  24, 24)
  → Block 3: Conv(64→128)+ BN + ReLU + MaxPool  → (B, 128, 12, 12)
  → Block 4: Conv(128→256)+BN + ReLU + MaxPool  → (B, 256,  6,  6)
  → Global Average Pool                          → (B, 256)
  → Dropout(0.4)
  → Linear(256 → 10)                             → (B, 10)
```

- No pretrained weights. No `torchvision.models`. Built from scratch.
- **Weighted cross-entropy** handles the class imbalance (class 4 has only 333 samples).
- **GPU augmentation** (random flip + rot90) applied per batch directly in the training loop — no DataLoader workers needed (avoids Windows multiprocessing issues).

---

## Ablation Study — Hyperparameters to Vary

| Parameter | File | Default | What to try |
|---|---|---|---|
| `IMG_SIZE` | `prepare_data.py` | `96` | `128` |
| `USE_BN` | `model.py` | `True` | `False` |
| `DROPOUT_P` | `model.py` | `0.4` | `0.0`, `0.2`, `0.6` |
| `USE_AUG` | `train.py` | `True` | `False` |
| `LR` | `train.py` | `1e-3` | `5e-4`, `3e-3` |
| `BATCH_SIZE` | `train.py` | `64` | `32`, `128` |
| `EPOCHS` | `train.py` | `50` | `100` |
| `PATIENCE` | `train.py` | `10` | `15` |

Re-run `prepare_data.py` only if you change `IMG_SIZE`. All other flags only require re-running `train.py` (and then `evaluate.py`).

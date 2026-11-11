"""
train.py
--------
Step 4: Training loop only.

What this file does:
  - Loads train.npz and val.npz from the data/ folder
  - Applies GPU augmentation (flip + rot90) on each training batch
  - Uses AdamW optimiser + OneCycleLR scheduler
  - Weighted cross-entropy to handle class imbalance
  - Early stopping on validation accuracy (patience = PATIENCE epochs)
  - Saves the best model weights to outputs/best_model.pth
  - Saves loss and accuracy curves to outputs/

Ablation flags at the top of the file:
    EPOCHS, LR, BATCH_SIZE, USE_AUG, PATIENCE

Run:
    D:\Gpu-shit\nasa\Scripts\python.exe train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, random

from model import GalaxyCNN

# ── Ablation / hyperparameter flags ─────────────────────────────────────────
EPOCHS      = 50        # max training epochs
LR          = 1e-3      # initial learning rate for AdamW
BATCH_SIZE  = 64        # images per batch
USE_AUG     = True      # set False to ablate GPU augmentation
PATIENCE    = 10        # early-stop if val accuracy doesn't improve for N epochs
SEED        = 42
DATA_DIR    = "data"
OUT_DIR     = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Makes conv ops deterministic (small speed cost, worth it for reproducibility)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("WARNING: CUDA not available — training will be very slow on CPU.")

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
train_data = np.load(os.path.join(DATA_DIR, "train.npz"))
val_data   = np.load(os.path.join(DATA_DIR, "val.npz"))

# Images are stored as uint8 (0-255, CHW). Convert to float32 [0,1] here.
X_train = train_data["images"].astype(np.float32) / 255.0   # (N, 3, H, W)
y_train = train_data["labels"].astype(np.int64)

X_val   = val_data["images"].astype(np.float32) / 255.0
y_val   = val_data["labels"].astype(np.int64)

# Mean and std were computed from the train set in prepare_data.py
mean = train_data["mean"].astype(np.float32)   # shape (3,)
std  = train_data["std"].astype(np.float32)    # shape (3,)

# ── 2. Normalise ─────────────────────────────────────────────────────────────
# Reshape mean/std to (1, 3, 1, 1) for broadcasting over (N, C, H, W)
mean_t = torch.tensor(mean).view(1, 3, 1, 1)
std_t  = torch.tensor(std).view(1, 3, 1, 1)

X_train_t = (torch.from_numpy(X_train) - mean_t) / std_t
X_val_t   = (torch.from_numpy(X_val)   - mean_t) / std_t
y_train_t = torch.from_numpy(y_train)
y_val_t   = torch.from_numpy(y_val)

# ── 3. DataLoaders ───────────────────────────────────────────────────────────
# pin_memory=True speeds up CPU→GPU transfer
# num_workers=0 avoids multiprocessing issues on Windows
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=(device.type == "cuda"), num_workers=0
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=(device.type == "cuda"), num_workers=0
)

# ── 4. Class weights for imbalanced cross-entropy ────────────────────────────
# Formula: weight_c = N / (num_classes * count_c)
# Rarer classes get a higher weight so their loss contribution is amplified.
counts = np.bincount(y_train, minlength=10)
weights = len(y_train) / (10.0 * counts)          # shape (10,)
weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
print(f"Class weights: {np.round(weights, 3)}")

# ── 5. Model, loss, optimiser, scheduler ────────────────────────────────────
model     = GalaxyCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=weights_t)
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# OneCycleLR: ramps LR up then down over training — often beats step/cosine
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)

# ── 6. GPU augmentation function ─────────────────────────────────────────────
def augment(x):
    """
    Apply random flips and 90-degree rotations on a GPU batch tensor.
    Galaxy orientation doesn't affect class, so these are safe transforms.
    x shape: (B, C, H, W) — already on GPU.
    """
    # Random horizontal flip (50% chance per batch)
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[3])   # flip width dimension
    # Random vertical flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[2])   # flip height dimension
    # Random 90-degree rotation: k ∈ {0, 1, 2, 3} → 0°, 90°, 180°, 270°
    k = torch.randint(0, 4, (1,)).item()
    x = torch.rot90(x, k=k, dims=[2, 3])
    return x

# ── 7. Training loop ─────────────────────────────────────────────────────────
best_val_acc   = 0.0
epochs_no_improve = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print(f"\nStarting training for up to {EPOCHS} epochs (early stop patience={PATIENCE})...")
print(f"Augmentation: {'ON' if USE_AUG else 'OFF'}\n")

for epoch in range(EPOCHS):

    # ── Train phase ──────────────────────────────────────────────────────────
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        if USE_AUG:
            xb = augment(xb)   # augment on GPU in-place

        optimiser.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimiser.step()
        scheduler.step()       # OneCycleLR steps every batch, not every epoch

        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += len(yb)

    train_loss = total_loss / total
    train_acc  = correct / total

    # ── Validation phase ─────────────────────────────────────────────────────
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            v_loss += criterion(logits, yb).item() * len(yb)
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total   += len(yb)

    val_loss = v_loss / v_total
    val_acc  = v_correct / v_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f}  acc {val_acc:.4f}")

    # ── Save best model ───────────────────────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
        print(f"  ✓ New best val acc {best_val_acc:.4f} — model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

# ── 8. Plot and save curves ───────────────────────────────────────────────────
epochs_ran = len(history["train_loss"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="train")
ax1.plot(history["val_loss"],   label="val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

ax2.plot(history["train_acc"], label="train")
ax2.plot(history["val_acc"],   label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "curves.png"), dpi=150)
plt.close()

print(f"\nBest val accuracy: {best_val_acc:.4f}")
print(f"Curves saved → {OUT_DIR}/curves.png")
print("Run evaluate.py next.")

"""
evaluate.py
-----------
Step 5: One-time evaluation on the held-out TEST set.

This file is run ONCE after training is complete. It:
  - Loads the best saved model weights
  - Evaluates on test.npz (never seen during training or early stopping)
  - Prints overall accuracy + per-class precision/recall/F1
  - Saves a confusion matrix plot → outputs/confusion_matrix.png
  - Saves a grid of misclassified examples → outputs/misclassified.png

Run:
    D:\Gpu-shit\nasa\Scripts\python.exe evaluate.py
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

from model import GalaxyCNN

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
OUT_DIR     = "outputs"
MODEL_PATH  = os.path.join(OUT_DIR, "best_model.pth")
BATCH_SIZE  = 64

CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round smooth",
    "In-between smooth",
    "Cigar-shaped",
    "Barred spiral",
    "Unbarred tight",
    "Unbarred loose",
    "Edge-on no bulge",
    "Edge-on bulge",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 1. Load test data ────────────────────────────────────────────────────────
print("Loading test data...")
test_data  = np.load(os.path.join(DATA_DIR, "test.npz"))
train_data = np.load(os.path.join(DATA_DIR, "train.npz"))   # for mean/std only

X_test = test_data["images"].astype(np.float32) / 255.0
y_test = test_data["labels"].astype(np.int64)

# Normalise with TRAIN mean/std (same as during training — no leakage)
mean = train_data["mean"].astype(np.float32)
std  = train_data["std"].astype(np.float32)
mean_t = torch.tensor(mean).view(1, 3, 1, 1)
std_t  = torch.tensor(std).view(1, 3, 1, 1)

X_test_t = (torch.from_numpy(X_test) - mean_t) / std_t
y_test_t  = torch.from_numpy(y_test)

test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ── 2. Load model ────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
model = GalaxyCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# ── 3. Run inference ─────────────────────────────────────────────────────────
all_preds, all_true = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

acc = (all_preds == all_true).mean()
print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

# ── 4. Classification report (precision / recall / F1 per class) ─────────────
print("Per-class report:")
print(classification_report(all_true, all_preds, target_names=CLASS_NAMES, digits=3))

# ── 5. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(all_true, all_preds)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Test Set")

# Annotate each cell with its count
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"Confusion matrix saved → {OUT_DIR}/confusion_matrix.png")

# ── 6. Misclassified examples grid ───────────────────────────────────────────
# Find indices where the model was wrong, show up to 20 examples
wrong_idx = np.where(all_preds != all_true)[0]
print(f"\nMisclassified: {len(wrong_idx)} / {len(all_true)}  "
      f"({len(wrong_idx)/len(all_true)*100:.1f}%)")

n_show = min(20, len(wrong_idx))
fig, axes = plt.subplots(4, 5, figsize=(14, 11))
axes = axes.flatten()

# De-normalise images for display: reverse (x - mean) / std → x * std + mean
mean_disp = torch.tensor(mean).view(3, 1, 1)
std_disp  = torch.tensor(std).view(3, 1, 1)

for i in range(n_show):
    wi  = wrong_idx[i]                        # index in test set
    img = X_test_t[wi] * std_disp + mean_disp # de-normalise
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()   # CHW → HWC, clip to [0,1]

    axes[i].imshow(img)
    axes[i].set_title(
        f"True: {CLASS_NAMES[all_true[wi]]}\nPred: {CLASS_NAMES[all_preds[wi]]}",
        fontsize=7, color="red"
    )
    axes[i].axis("off")

# Hide any unused subplots
for i in range(n_show, len(axes)):
    axes[i].axis("off")

plt.suptitle("Misclassified Examples", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "misclassified.png"), dpi=150)
plt.close()
print(f"Misclassified grid saved → {OUT_DIR}/misclassified.png")
print("\nEvaluation complete.")

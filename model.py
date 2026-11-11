"""
model.py
--------
Step 3: The CNN architecture — nothing else.

Design: 4 conv blocks, each with Conv → (optional BN) → ReLU → MaxPool.
        Global Average Pooling collapses the spatial dims without a huge
        fully-connected layer, keeping the param count low and helping
        with generalisation.
        A single Dropout + Linear layer forms the classification head.

Ablation flags (change these to run experiments):
    USE_BN      = True/False   toggle Batch Normalisation
    DROPOUT_P   = 0.0–0.9      dropout probability (0.0 = off)

No pretrained weights. No torchvision.models. Built from scratch.
"""

import torch
import torch.nn as nn

# ── Ablation flags ───────────────────────────────────────────────────────────
USE_BN    = True     # set False to ablate Batch Normalisation
DROPOUT_P = 0.4      # set 0.0 to ablate Dropout


def conv_block(in_ch, out_ch, use_bn=USE_BN):
    """
    One convolutional block:
        Conv2d(3×3, same padding) → (BatchNorm) → ReLU → MaxPool(2×2)

    padding=1 keeps spatial size the same through conv;
    MaxPool(2) then halves H and W.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
        # bias=False when BN follows — BN already shifts, so bias is redundant
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # halve spatial dims
    return nn.Sequential(*layers)


class GalaxyCNN(nn.Module):
    """
    Input : (B, 3, 96, 96)   — 3-channel RGB, 96×96
    Output: (B, 10)           — raw logits for 10 galaxy classes

    Spatial progression (with 96×96 input):
        After block 1 → (B, 32,  48, 48)
        After block 2 → (B, 64,  24, 24)
        After block 3 → (B, 128, 12, 12)
        After block 4 → (B, 256,  6,  6)
        After GAP     → (B, 256)
        After head    → (B, 10)
    """

    def __init__(self, num_classes=10, use_bn=USE_BN, dropout_p=DROPOUT_P):
        super().__init__()

        # Four convolutional blocks, doubling channels each time
        self.block1 = conv_block(3,   32,  use_bn)
        self.block2 = conv_block(32,  64,  use_bn)
        self.block3 = conv_block(64,  128, use_bn)
        self.block4 = conv_block(128, 256, use_bn)

        # Global Average Pooling: averages over H and W → (B, 256)
        # This replaces a large Flatten + Linear layer, reducing overfitting.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)          # (B, 256, 1, 1)
        x = x.flatten(1)         # (B, 256)
        x = self.dropout(x)
        x = self.fc(x)           # (B, 10) — raw logits
        return x


# ── Quick sanity check (run this file directly to verify shapes) ─────────────
if __name__ == "__main__":
    model = GalaxyCNN()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Fake batch to verify forward pass dimensions
    dummy = torch.zeros(4, 3, 96, 96)
    out   = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}   ← should be (4, 10)")

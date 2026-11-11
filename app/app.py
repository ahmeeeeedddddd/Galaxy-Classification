"""
app/app.py
----------
A very simple web interface to:
1. Upload any galaxy image and get predictions + confidence probabilities.
2. View model metrics, curves, and confusion matrix on a separate stats page.

Run:
    D:\Gpu-shit\nasa\Scripts\python.exe app/app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

# Add parent directory to path so we can import model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import GalaxyCNN

app = Flask(__name__)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

# Load model and normalisation statistics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

def load_resources():
    global model, mean, std
    # Load mean / std if prepared
    train_npz = os.path.join(DATA_DIR, "train.npz")
    if os.path.exists(train_npz):
        try:
            data = np.load(train_npz)
            mean = data["mean"].astype(np.float32)
            std = data["std"].astype(np.float32)
        except Exception as e:
            print(f"Warning: could not read mean/std from train.npz: {e}")

    # Load model weights if saved
    model_path = os.path.join(OUTPUTS_DIR, "best_model.pth")
    model = GalaxyCNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print("Warning: best_model.pth not found. Model will use random weights until trained.")
    model.eval()

load_resources()

def preprocess_image(image_path):
    """Resize to 96x96, convert to tensor, and normalise using train mean/std."""
    pil_img = Image.open(image_path).convert("RGB").resize((96, 96), Image.BILINEAR)
    arr = np.array(pil_img).astype(np.float32) / 255.0  # H, W, C
    arr = arr.transpose(2, 0, 1)                        # C, H, W
    
    # Normalise
    for c in range(3):
        arr[c] = (arr[c] - mean[c]) / std[c]
        
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 3, 96, 96)
    return tensor

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # Predict
            tensor = preprocess_image(save_path)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            pred_class_idx = int(np.argmax(probs))
            pred_class_name = CLASS_NAMES[pred_class_idx]
            confidence = float(probs[pred_class_idx]) * 100.0

            # Package all 10 probabilities for display
            all_probs = [
                {"name": CLASS_NAMES[i], "prob": round(float(probs[i]) * 100.0, 2)}
                for i in range(10)
            ]
            all_probs.sort(key=lambda x: x["prob"], reverse=True)

            return render_template(
                "index.html",
                filename=filename,
                pred_class=pred_class_name,
                confidence=round(confidence, 2),
                all_probs=all_probs,
            )

    return render_template("index.html")

@app.route("/stats")
def stats():
    # Check which output files exist
    curves_exist = os.path.exists(os.path.join(OUTPUTS_DIR, "curves.png"))
    cm_exist = os.path.exists(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
    misc_exist = os.path.exists(os.path.join(OUTPUTS_DIR, "misclassified.png"))
    grid_exist = os.path.exists(os.path.join(OUTPUTS_DIR, "class_grid.png"))

    return render_template(
        "stats.html",
        curves_exist=curves_exist,
        cm_exist=cm_exist,
        misc_exist=misc_exist,
        grid_exist=grid_exist,
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

if __name__ == "__main__":
    print("\nStarting Galaxy Web App...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)

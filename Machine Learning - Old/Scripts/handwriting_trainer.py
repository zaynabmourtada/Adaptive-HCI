#!/usr/bin/env python3

"""
handwriting_trainer.py

Prompts user for:
  - CSV file path containing columns [X, Y, Frame].
  - Letter label (A–Z).
Loads or creates a model in ~/xamera_machine_learning/models/handwriting_lstm_model.pth,
trains it (toy example), then saves/updates the model.

Appends training info to a summary CSV in:
  ~/xamera_machine_learning/results/trained_model_summary.csv

Usage:
  $ python3 handwriting_trainer.py
"""

import os
import csv
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# A–Z letter dictionary
# ---------------------------
LETTER_TO_ID = {
    letter: idx for idx, letter in enumerate(string.ascii_uppercase)
    # This maps 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25
}

# ---------------------------
# 1) Prompt user for CSV + letter
# ---------------------------
csv_path = input("Enter the path to your handwriting CSV (with columns X,Y,Frame): ").strip()
letter_label = input("Which letter does this CSV represent? (A–Z): ").strip().upper()

# Expand ~ if present
csv_path = os.path.expanduser(csv_path)

# Validate letter
if letter_label not in LETTER_TO_ID:
    print(f"[ERROR] Letter '{letter_label}' not in A–Z. Exiting.")
    exit(1)

# Convert letter to numeric ID
label_id = LETTER_TO_ID[letter_label]

# ---------------------------
# 2) Load CSV data
# ---------------------------
# We'll skip any lines that contain error logs, 'Error processing frame', etc.
# By default, we expect a header row: X,Y,Frame
try:
    # Attempt reading with a header row named 'X','Y','Frame'.
    df = pd.read_csv(csv_path)
except:
    print(f"[ERROR] Could not read CSV at {csv_path}. Exiting.")
    exit(1)

# Basic check: ensure columns are [X, Y, Frame]
if not set(['X', 'Y', 'Frame']).issubset(df.columns):
    print(f"[ERROR] CSV must have columns X, Y, Frame. Found columns: {list(df.columns)}")
    exit(1)

# Drop any rows that contain strings like "Error" or "System.err"
df = df[~df['X'].astype(str).str.contains("Error|System|D ")]
df = df[~df['Y'].astype(str).str.contains("Error|System|D ")]
df = df[~df['Frame'].astype(str).str.contains("Error|System|D ")]

# Convert to float
try:
    coords_np = df[['X', 'Y', 'Frame']].astype(float).to_numpy()
except:
    print("[ERROR] Could not convert X,Y,Frame to float. Possibly invalid rows.")
    exit(1)

# coords_np shape => (num_points, 3)
print(f"[INFO] Loaded {coords_np.shape[0]} coordinate rows from {csv_path}.")

# ---------------------------
# 3) Prepare Data + Label
# ---------------------------
# For a toy approach, treat the entire coordinate set as one "sample".
# We'll make a single input vector by flattening or just do an average, etc.
# Example: We'll just compute the mean of (X, Y, Frame). This is extremely naive.
# A real approach might feed the sequence to an LSTM, or do more advanced feature extraction.
mean_coords = coords_np.mean(axis=0)  # shape => (3,)
mean_coords_torch = torch.tensor(mean_coords, dtype=torch.float32).unsqueeze(0)
# We'll store the label as a single integer
label_torch = torch.tensor([label_id], dtype=torch.long)

# ---------------------------
# 4) Simple Model Definition
# ---------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 26)  # 26 letters
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# 5) Load or Create Model
# ---------------------------
model_dir = os.path.expanduser("~/xamera_machine_learning/models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "handwriting_lstm_model.pth")

model = SimpleModel()
if os.path.isfile(model_path):
    print(f"[INFO] Found existing model at: {model_path}. Loading...")
    model.load_state_dict(torch.load(model_path))
else:
    print(f"[INFO] No existing model at: {model_path}. Creating a new one...")

# ---------------------------
# 6) Training Setup
# ---------------------------
criterion = nn.CrossEntropyLoss()  # for classification among 26 letters
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# We'll do a small number of epochs for demonstration
EPOCHS = 10

print(f"\nStarting training on CSV: {csv_path}")
print(f"Label: {letter_label}")

x_train = mean_coords_torch  # shape (1,3)
y_train = label_torch         # shape (1,)

# ---------------------------
# 7) Train
# ---------------------------
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(x_train)         # shape => (1,26)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1 == 0:  # print every epoch
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

print("\nTraining complete.")

# ---------------------------
# 8) Save Model
# ---------------------------
torch.save(model.state_dict(), model_path)
print(f"[INFO] Model saved to: {model_path}")

# ---------------------------
# 9) Evaluate Quick Prediction
# ---------------------------
model.eval()
with torch.no_grad():
    logits = model(x_train)     # shape (1,26)
    predicted_id = int(logits.argmax(dim=1).item())
    # Reverse map
    id_to_letter = {v: k for k, v in LETTER_TO_ID.items()}
    recognized_letter = id_to_letter.get(predicted_id, "?")

print(f"\nPredicted numeric ID: {predicted_id}")
print(f"Recognized letter: {recognized_letter}")

# ---------------------------
# 10) Append summary
# ---------------------------
results_dir = os.path.expanduser("~/xamera_machine_learning/results")
os.makedirs(results_dir, exist_ok=True)
summary_csv = os.path.join(results_dir, "trained_model_summary.csv")

file_exists = os.path.isfile(summary_csv)
with open(summary_csv, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["CSV_File", "LabelEntered", "PredictedID", "PredictedLetter"])
    writer.writerow([csv_path, letter_label, predicted_id, recognized_letter])

print(f"[INFO] Summary appended to: {summary_csv}")
print("[DONE]")

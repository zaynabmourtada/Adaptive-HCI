#!/usr/bin/env python3

"""
digits_trainer.py

Single-file script to train a digit classification model (digits 0–9)
from coordinate CSV data. It:

  - Prompts user for CSV path (columns [X, Y, Frame]).
  - Prompts user for which digit (0–9) the CSV represents.
  - Cleans CSV lines with "Error", etc.
  - Builds/trains an LSTM-based classification model (toy example).
  - Saves model to ~/xamera_machine_learning/models/digits_lstm_model.pth
  - Appends results to ~/xamera_machine_learning/results/digits_training_summary.csv

Usage:
  $ python3 digits_trainer.py
"""

import os
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# DIGIT_TO_ID: map '0'..'9' -> 0..9
# ---------------------------
DIGIT_TO_ID = {str(d): d for d in range(10)}

# ---------------------------
# LSTM-based Model Definition
# ---------------------------
class DigitsLSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_digits=10):
        super(DigitsLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_digits)  # output size => 10 classes

    def forward(self, x):
        """
        x => shape: (batch_size, seq_len, input_dim)
        Returns logits => (batch_size, num_digits).
        """
        lstm_out, (hn, cn) = self.lstm(x)
        # Take last time-step
        last_step = lstm_out[:, -1, :]  # => (batch_size, hidden_dim)
        logits = self.fc(last_step)     # => (batch_size, num_digits)
        return logits

def main():
    # 1) Prompt user
    csv_path = input("Enter the path to your handwriting CSV (columns [X,Y,Frame]): ").strip()
    digit_label = input("Which digit does this CSV represent? (0–9): ").strip()

    # Expand user path
    csv_path = os.path.expanduser(csv_path)

    # Validate digit
    if digit_label not in DIGIT_TO_ID:
        print(f"[ERROR] '{digit_label}' not in '0'..'9'. Exiting.")
        return

    label_id = DIGIT_TO_ID[digit_label]

    # 2) Load CSV data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV at {csv_path}: {e}")
        return

    # Ensure columns
    required_cols = {"X", "Y", "Frame"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] CSV must have columns X, Y, Frame. Found: {list(df.columns)}")
        return

    # Drop lines with “Error” or log content
    for col in ["X", "Y", "Frame"]:
        df = df[~df[col].astype(str).str.contains("Error|System|D ")]
        # You can also exclude lines with “E  Error processing frame” etc. if needed.

    # Convert to float
    try:
        coords_np = df[["X", "Y", "Frame"]].astype(float).to_numpy()
    except ValueError as ve:
        print(f"[ERROR] Failed converting columns to float: {ve}")
        return

    num_points = coords_np.shape[0]
    print(f"[INFO] Loaded {num_points} coordinate rows from {csv_path} for digit '{digit_label}'.")

    # 3) Prepare data
    # We treat the entire coordinate set as a single sequence => shape (1, seq_len, 3).
    coords_tensor = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0)
    label_tensor = torch.tensor([label_id], dtype=torch.long)  # => shape (1,)

    # 4) Create or Load Model
    model_dir = os.path.expanduser("~/xamera_machine_learning/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "digits_lstm_model.pth")

    # Instantiate the model
    model = DigitsLSTMModel(input_dim=3, hidden_dim=32, num_digits=10)
    if os.path.isfile(model_path):
        print(f"[INFO] Found existing model at: {model_path}. Loading weights...")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"[INFO] No existing model at: {model_path}. Creating a new one...")

    # 5) Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10
    print(f"\nStarting training on CSV: {csv_path}")
    print(f"Label/digit: {digit_label}")

    # 6) Train
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(coords_tensor)          # => shape (1,10)
        loss = criterion(logits, label_tensor) # single-sample
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

    print("\nTraining complete.")

    # 7) Save Model
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to: {model_path}")

    # 8) Quick Evaluate
    model.eval()
    with torch.no_grad():
        out = model(coords_tensor)   # => (1,10)
        pred_id = int(out.argmax(dim=1).item())
    print(f"Predicted numeric ID: {pred_id}")

    # 9) Append summary
    results_dir = os.path.expanduser("~/xamera_machine_learning/results")
    os.makedirs(results_dir, exist_ok=True)
    summary_csv = os.path.join(results_dir, "digits_training_summary.csv")

    file_exists = os.path.isfile(summary_csv)
    with open(summary_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["CSV_File", "DigitLabel", "PredictedID"])
        writer.writerow([csv_path, digit_label, pred_id])

    print(f"[INFO] Summary appended to: {summary_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()

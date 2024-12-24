#!/usr/bin/env python3
"""
Handwriting Trainer (LSTM) for A–Z
----------------------------------
- Prompts user for:
    1) CSV path with columns [X, Y, Frame]
    2) Single letter label (A–Z)
- Loads or creates a classification LSTM model:
    input_size=3, hidden_size=32, num_classes=26
- Uses cross-entropy classification on a single-sequence "batch"
  (toy approach).
- Saves model to:
    ~/xamera_machine_learning/models/handwriting_lstm_model.pth
- Appends summary row to:
    ~/xamera_machine_learning/results/trained_model_summary.csv
  ^-- (Note the results folder change)

Usage:
  python3 handwriting_trainer.py
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

####################################################
# 1) Dictionary for letters A–Z => numeric IDs 0..25
####################################################
LETTER_TO_ID = {chr(i): (i - ord('A')) for i in range(ord('A'), ord('Z') + 1)}
ID_TO_LETTER = {v: k for k, v in LETTER_TO_ID.items()}

####################################################
# 2) LSTM Model (Classification)
####################################################
class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_classes=26):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)  # 26-class output

    def forward(self, x):
        """
        x shape: (batch=1, seq_len, 3)
        returns logits shape: (batch=1, 26)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # get last LSTM output
        last_step = lstm_out[:, -1, :]   # (1, hidden_dim)
        logits = self.fc(last_step)      # (1, 26)
        return logits

####################################################
# 3) Main training routine
####################################################
def main():
    # Prompt user for CSV path & letter
    csv_path = input("Enter the path to your CSV file (with header X,Y,Frame): ").strip()
    letter_str = input("Which letter does this CSV represent? (A–Z): ").strip().upper()

    # Validate letter
    if letter_str not in LETTER_TO_ID:
        raise ValueError(f"Letter must be A–Z. Got: {letter_str}")

    label_id = LETTER_TO_ID[letter_str]

    # Expand user path
    csv_path = os.path.expanduser(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Read CSV data
    # We assume the header is X,Y,Frame in row0, numeric rows follow
    df = pd.read_csv(csv_path, header=0)
    # Convert to float32
    coords_np = df[["X", "Y", "Frame"]].to_numpy(dtype="float32")

    # shape => (seq_len, 3). Then unsqueeze => (1, seq_len, 3) for single "batch"
    coords_tensor = torch.from_numpy(coords_np).unsqueeze(0)  # (1, seq_len, 3)
    # Build label tensor => single integer class (e.g. 'O' => 14)
    label_tensor = torch.tensor([label_id], dtype=torch.long)  # (1,)

    # Prepare directories
    model_dir = os.path.expanduser("~/xamera_machine_learning/models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "handwriting_lstm_model.pth")

    # ---- CHANGE: summary CSV is now in a "results" folder ----
    result_dir = os.path.expanduser("~/xamera_machine_learning/results")
    os.makedirs(result_dir, exist_ok=True)
    summary_csv = os.path.join(result_dir, "trained_model_summary.csv")

    # Instantiate model
    model = LSTMModel(input_dim=3, hidden_dim=32, num_classes=26)

    # If existing model, load it
    if os.path.isfile(model_path):
        print(f"[INFO] Found existing model at: {model_path}. Loading weights...")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"[INFO] No existing model at: {model_path}. Creating a new one...")

    # Setup training
    criterion = nn.CrossEntropyLoss()  # classification
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10
    print(f"\nStarting training on CSV: {csv_path}\nLabel: {letter_str}\n")

    # Train loop
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # forward pass
        logits = model(coords_tensor)  # (1, 26)
        loss = criterion(logits, label_tensor)  # label_tensor => (1,)

        # backward
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

    print("\nTraining complete.")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to: {model_path}\n")

    # Quick test inference to see "predicted" letter
    model.eval()
    with torch.no_grad():
        test_logits = model(coords_tensor)   # (1, 26)
        predicted_class = torch.argmax(test_logits, dim=1).item()
        recognized_letter = ID_TO_LETTER.get(predicted_class, "?")

    print(f"Predicted numeric ID: {predicted_class}")
    print(f"Recognized letter: {recognized_letter}")

    # Append or create summary CSV in results folder
    file_exists = os.path.isfile(summary_csv)
    with open(summary_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "CSV_Path", "LabeledLetter", 
                "PredictedID", "PredictedLetter", 
                "ModelFile"
            ])
        writer.writerow([
            csv_path,
            letter_str,
            predicted_class,
            recognized_letter,
            os.path.basename(model_path)
        ])

    print(f"[INFO] Summary appended to: {summary_csv}")
    print("[DONE]")

if __name__ == "__main__":
    main()

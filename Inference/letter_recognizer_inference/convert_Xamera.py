import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Paths
xamera_root = "Xamera Letter Dataset"
output_csv_train = "xamera-letters-train.csv"
output_csv_test = "xamera-letters-test.csv"

# List all folders (A-Z)
folders = sorted(os.listdir(xamera_root))  # A, B, C, ..., Z

# Create label mapping: 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25
label_map = {folder: i for i, folder in enumerate(folders)}

# Data storage
data = []

# Process images
for folder in tqdm(folders, desc="Processing Xamera Data"):
    folder_path = os.path.join(xamera_root, folder)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image
            img_path = os.path.join(folder_path, filename)
            
            # Open image, convert to grayscale, resize to 28x28
            img = Image.open(img_path).convert("L").resize((28, 28))
            img_array = np.array(img, dtype=np.uint8).flatten()  # Keep values 0-255
            
            # Append label + pixel data
            label = label_map[folder]
            data.append([label] + img_array.tolist())

# Convert to DataFrame
df = pd.DataFrame(data)

# Shuffle dataset before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# **Split into train and test (80% train, 20% test)**
train_size = int(0.8 * len(df))
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Save to CSV
df_train.to_csv(output_csv_train, index=False, header=False)
df_test.to_csv(output_csv_test, index=False, header=False)

print(f"Xamera dataset converted! Train: {len(df_train)} samples, Test: {len(df_test)} samples.")

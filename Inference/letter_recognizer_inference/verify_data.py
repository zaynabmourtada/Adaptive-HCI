import pandas as pd

# Load and check shape
df_xamera = pd.read_csv("xamera-letters-train.csv", header=None)
print(f"Shape: {df_xamera.shape} (should be num_samples x 785)")

# Check first few rows
print(df_xamera.head())

# Check unique class labels (should be 0-25)
print("Unique Labels:", df_xamera[0].unique())

# Check pixel value range (should be 0-255)
print("Pixel Range:", df_xamera.iloc[:, 1:].values.min(), "-", df_xamera.iloc[:, 1:].values.max())

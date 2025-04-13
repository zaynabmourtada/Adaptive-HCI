import os
import torch
from ultralytics import YOLO

# Define paths as before.
script_dir = os.path.dirname(os.path.abspath(__file__))
pt_path = os.path.join(script_dir, "best.pt")

# Use your custom model for conversion
model = YOLO(pt_path)
model.export(format="tflite")

print("✅ All conversions completed.")
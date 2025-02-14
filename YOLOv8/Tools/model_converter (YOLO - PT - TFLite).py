import os
from ultralytics import YOLO

# Define the path to best.pt directly
pt_path = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\8\weights\best.pt"

# Ensure the file exists
if not os.path.exists(pt_path):
    raise FileNotFoundError(f"File '{pt_path}' not found.")
else:
    print("✅ Found best.pt.")

# Define the output file paths based on best.pt's location and name
root = os.path.dirname(pt_path)
base_name = os.path.splitext(os.path.basename(pt_path))[0]
tflite_fp32 = os.path.join(root, f"{base_name}_YOLO_fp32.tflite")
tflite_fp16 = os.path.join(root, f"{base_name}_YOLO_fp16.tflite")

print(f"🔄 Converting {pt_path} to TFLite formats...")

# Load YOLO model
model = YOLO(pt_path)

# Convert to FP32 TFLite (default)
if not os.path.exists(tflite_fp32):
    model.export(format="tflite")  # Exports to {base_name}.tflite in the same directory
    os.rename(os.path.join(root, f"{base_name}.tflite"), tflite_fp32)
    print(f"✅ FP32 Model Saved: {tflite_fp32}")
else:
    print(f"⏭️ FP32 Model already exists: {tflite_fp32}")

# Convert to FP16 TFLite
if not os.path.exists(tflite_fp16):
    model.export(format="tflite", half=True)
    os.rename(os.path.join(root, f"{base_name}.tflite"), tflite_fp16)
    print(f"✅ FP16 Model Saved: {tflite_fp16}")
else:
    print(f"⏭️ FP16 Model already exists: {tflite_fp16}")

print("✅ All conversions completed.")

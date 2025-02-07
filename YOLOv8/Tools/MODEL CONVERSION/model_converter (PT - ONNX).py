import os
from ultralytics import YOLO

# Define input directory containing .pt files
input_dir = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights"

# Ensure the directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Directory '{input_dir}' not found.")

# Loop through all .pt files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".pt"):
        pt_path = os.path.join(input_dir, filename)
        onnx_path = os.path.join(input_dir, os.path.splitext(filename)[0] + ".onnx")  # Keep name, change extension
        
        # Skip if the ONNX file already exists
        if os.path.exists(onnx_path):
            print(f"Skipping {filename} (ONNX file already exists)")
            continue

        print(f"Converting {pt_path} to {onnx_path}...")

        # Load YOLO model
        model = YOLO(pt_path)

        # Export model to ONNX format
        model.export(format="onnx", dynamic=False, simplify=True)

        print(f"Successfully converted: {onnx_path}")

print("All conversions completed.")
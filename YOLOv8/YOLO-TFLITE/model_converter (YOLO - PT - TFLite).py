import os
from ultralytics import YOLO

# Define input directory containing .pt files
input_dir = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\4\weights"

# Ensure the directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Directory '{input_dir}' not found.")

# Get all .pt files
pt_files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]

if not pt_files:
    print("❌ No .pt files found in the directory!")
else:
    print("✅ Found .pt files:", pt_files)

# Loop through all .pt files (including subdirectories)
for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".pt"):
            pt_path = os.path.join(root, filename)

            # Define TFLite output file paths
            base_name = os.path.splitext(filename)[0]
            tflite_fp32 = os.path.join(root, f"{base_name}_YOLO_fp32.tflite")
            tflite_fp16 = os.path.join(root, f"{base_name}_YOLO_fp16.tflite")
            tflite_int8 = os.path.join(root, f"{base_name}_YOLO_int8.tflite")

            # Skip conversion if TFLite files already exist
            if os.path.exists(tflite_fp32) and os.path.exists(tflite_fp16) and os.path.exists(tflite_int8):
                print(f"⏭️ Skipping {filename} (All TFLite models already exist)")
                continue

            print(f"🔄 Converting {pt_path} to TFLite formats...")

            # Load YOLO model
            model = YOLO(pt_path)

            # Convert to FP32 TFLite
            if not os.path.exists(tflite_fp32):
                model.export(format="tflite")  # Default is FP32
                os.rename(os.path.join(root, f"{base_name}.tflite"), tflite_fp32)
                print(f"✅ FP32 Model Saved: {tflite_fp32}")

            # Convert to FP16 TFLite
            if not os.path.exists(tflite_fp16):
                model.export(format="tflite", half=True)  # Enables FP16
                os.rename(os.path.join(root, f"{base_name}.tflite"), tflite_fp16)
                print(f"✅ FP16 Model Saved: {tflite_fp16}")

            # Attempt INT8 Quantization (Only works if YOLO supports it)
            if not os.path.exists(tflite_int8):
                try:
                    model.export(format="tflite", int8=True)  # INT8 Quantization (Experimental)
                    os.rename(os.path.join(root, f"{base_name}.tflite"), tflite_int8)
                    print(f"✅ INT8 Model Saved: {tflite_int8}")
                except Exception as e:
                    print(f"⚠️ INT8 Quantization failed for {pt_path}: {e}")

print("✅ All conversions completed.")

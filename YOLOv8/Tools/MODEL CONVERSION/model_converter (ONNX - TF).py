import os
import onnx
import tf_keras
from onnx2tf import convert

# Define the input directory where .onnx models are stored
input_dir = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights"

# Ensure the directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Directory '{input_dir}' not found.")

# Loop through all .onnx files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".onnx"):
        onnx_path = os.path.join(input_dir, filename)
        tf_output_dir = os.path.join(input_dir, os.path.splitext(filename)[0] + "_tf")  # Keep separate folder

        # Skip if the TensorFlow model directory already exists
        if os.path.exists(tf_output_dir):
            print(f"Skipping {filename} (TensorFlow model already exists)")
            continue

        print(f"Converting {onnx_path} to TensorFlow SavedModel at {tf_output_dir}...")

        # Convert ONNX to TensorFlow SavedModel format
        convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tf_output_dir,
            copy_onnx_input_output_names_to_tflite=True
        )

        print(f"Successfully converted {onnx_path} to TensorFlow at {tf_output_dir}")

print("All ONNX to TensorFlow conversions completed.")
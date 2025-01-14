import os
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from ultralytics import YOLO

def process_files(input_dir):
    # Walk through all files and directories in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pt"):  # Process only .pt files
                input_file_path = os.path.join(root, file)

                try:
                    print(f"Processing file: {input_file_path}")
                    model = YOLO(input_file_path)

                    # Export the model to TorchScript format
                    export_dir = os.path.dirname(input_file_path)
                    model.export(
                        format="torchscript",
                        device="cpu",
                        imgsz=640,  # Adjust input size if needed
                        save_dir=export_dir  # Save TorchScript model in the same directory
                    )
                    
                    # Locate the exported TorchScript model
                    exported_model_path = os.path.join(export_dir, "best.torchscript")
                    if not os.path.exists(exported_model_path):
                        exported_model_path = os.path.join(export_dir, "last.torchscript")
                    
                    if not os.path.exists(exported_model_path):
                        raise FileNotFoundError("TorchScript model not found after export.")

                    # Optimize the exported TorchScript model for mobile
                    print(f"Optimizing model: {exported_model_path}")
                    scripted_model = torch.jit.load(exported_model_path)
                    optimized_model = optimize_for_mobile(scripted_model)

                    # Save the optimized model in the same directory
                    optimized_model_path = exported_model_path.replace(".torchscript", "_optimized.torchscript")
                    optimized_model.save(optimized_model_path)

                    print(f"Saved optimized model to: {optimized_model_path}")
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")

def main():
    # Hardcoded input directory
    input_dir = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights"

    # Process all files in the input directory
    process_files(input_dir)

if __name__ == "__main__":
    main()

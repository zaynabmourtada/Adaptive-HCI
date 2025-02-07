import os
from ultralytics import YOLO

def process_files(input_dir):
    # Walk through all files and directories in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pt"):  # Process only .pt files
                input_file_path = os.path.join(root, file)

                try:
                    print(f"🔄 Processing model: {input_file_path}")
                    model = YOLO(input_file_path)

                    # Export the model to TorchScript format
                    export_dir = os.path.dirname(input_file_path)
                    model.export(
                        format="torchscript",
                        device="cpu",
                        imgsz=960,  # Adjust input size if needed
                        save_dir=export_dir
                    )

                    # Locate the exported model
                    exported_model_path = os.path.join(export_dir, "best.torchscript")
                    if not os.path.exists(exported_model_path):
                        exported_model_path = os.path.join(export_dir, "last.torchscript")

                    if os.path.exists(exported_model_path):
                        print(f"✅ Model successfully exported: {exported_model_path}")
                    else:
                        print(f"❌ Warning: TorchScript model not found after export in {export_dir}!")

                except Exception as e:
                    print(f"❌ Error processing {input_file_path}: {e}")

def main():
    # Hardcoded input directory
    input_dir = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights"

    # Process all files in the input directory
    process_files(input_dir)

if __name__ == "__main__":
    main()
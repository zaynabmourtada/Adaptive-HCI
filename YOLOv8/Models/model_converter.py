import os
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
                    model.export(
                        format="torchscript",
                        device="cpu",
                        imgsz=640  # Adjust input size if needed
                    )
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")

def main():
    input_dir = "input"  # Directory containing the .pt files

    # Process all files in the input directory
    process_files(input_dir)

if __name__ == "__main__":
    main()

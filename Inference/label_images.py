import os
import shutil

# Define dataset path
dataset_path = "C:/Users/zayna/OneDrive/Documents/University/Senior Design/adaptive_code/Adaptive-HCI/Inference/Xamera Dataset"
output_path = os.path.join(dataset_path, "Sorted_Dataset")  # New structured dataset folder

# Create digit folders (0-9) if they don't exist
for i in range(10):
    digit_folder = os.path.join(output_path, str(i))
    os.makedirs(digit_folder, exist_ok=True)

# Move and rename images
for img_file in os.listdir(dataset_path):
    if img_file.endswith(".png"):
        digit_label = input(f"Enter the digit for {img_file}: ")  # Ask user for confirmation
        
        # Define destination folder
        dest_folder = os.path.join(output_path, digit_label)
        os.makedirs(dest_folder, exist_ok=True)  # Ensure folder exists
        
        # Move image to correct folder
        src_path = os.path.join(dataset_path, img_file)
        dest_path = os.path.join(dest_folder, img_file)  # Keep original filename
        
        shutil.move(src_path, dest_path)
        print(f"Moved {img_file} to {dest_folder}")

print(f"✅ Dataset sorted into folders like DIDA at: {output_path}")

# Author: Zaynab Mourtada 
# Purpose: Preprocess image data for inference model training
# Last Modified: 4/15/2025
import os
import cv2

def preprocess_images(
    input_folder,
    output_folder=None,
    rename_with_prefix=True,
    delete_original=False
):
    if not os.path.isdir(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return

    # Collect only image files
    files = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']
    ])

    if not files:
        print("No valid image files found in the folder.")
        return
    
    # If no output folder is provided, save in the same folder
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # Used for renaming files like '[folder_name]_1.png' based on folder name
    folder_prefix = os.path.basename(input_folder) if rename_with_prefix else ""

    for index, file_name in enumerate(files, start=1):
        old_path = os.path.join(input_folder, file_name)
        file_ext = os.path.splitext(file_name)[1].lower()

        new_name = f"{folder_prefix}_{index}{file_ext}" if rename_with_prefix else file_name
        new_path = os.path.join(output_folder, new_name)
        
        # Grayscale image
        img = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {old_path}")
            continue
        
        # Resize to 28x28
        resized_img = cv2.resize(img, (28, 28))
        cv2.imwrite(new_path, resized_img)
        
        # Delete original file if overwritten
        if delete_original and output_folder == input_folder and new_name != file_name:
            try:
                os.remove(old_path)
            except OSError as e:
                print(f"Error deleting {file_name}: {e}")

        if rename_with_prefix or output_folder != input_folder:
            print(f"Processed: {file_name} → {new_name}")
        else:
            print(f"Processed: {file_name}")

    print("\nPreprocessing is complete.")
    print(f"Output saved to: {output_folder}")


if __name__ == "__main__":
    input_dir = input("Enter input folder path: ").strip()
    out_dir = input("Output folder (leave blank to save in the same folder): ").strip() or None

    preprocess_images(
        input_folder=input_dir,
        output_folder=out_dir,
        rename_with_prefix=True,
        delete_original=False
    )

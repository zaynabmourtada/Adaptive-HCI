import os
import cv2

def preprocess_and_rename(folder_path):
    # Get a sorted list of files in the folder
    files = sorted(os.listdir(folder_path))

    # Loop through and rename files while processing
    for index, file_name in enumerate(files, start=1):
        file_ext = os.path.splitext(file_name)[1]  # Extract file extension

        # Skip non-image files
        if file_ext.lower() not in ['.png', '.jpg', '.jpeg']:
            print(f"Skipping non-image file: {file_name}")
            continue

        # Generate new file name (e.g., Z_1.png)
        folder_name = os.path.basename(folder_path)  # Get the last part of the path as the folder name
        new_name = f"{folder_name}_{index}{file_ext}"  # e.g., Z_1.png
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        # Read the image in grayscale mode
        img = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image was read successfully
        if img is None:
            print(f"Failed to read image: {old_path}")
            continue

        # Resize the image to 28x28
        resized_img = cv2.resize(img, (28, 28))

        # Save the processed image with the new name
        cv2.imwrite(new_path, resized_img)

        # If the new name differs from the old name, delete the old file
        if new_name != file_name:
            try:
                os.remove(old_path)
            except OSError as e:
                print(f"Error removing {old_path}: {e}")

        print(f"Renamed and processed: {file_name} -> {new_name}")

    print("Renaming and preprocessing complete!")

# Example usage
folder_path = input("Enter the folder path: ").strip()
if os.path.isdir(folder_path):
    preprocess_and_rename(folder_path)
else:
    print(f"Error: The folder '{folder_path}' does not exist.")

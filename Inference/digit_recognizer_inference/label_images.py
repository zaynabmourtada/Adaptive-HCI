import os
# Define the folder path
folder_path = "Xamera Dataset/9_converted"  # Change this to the folder containing images of 0s

# Get all files in the folder
files = sorted(os.listdir(folder_path))  # Sorting ensures order

# Loop through and rename files
for index, file_name in enumerate(files, start=1):
    file_ext = os.path.splitext(file_name)[1]  # Get the file extension
    new_name = f"9_{index}{file_ext}"  # Rename format: 0_1, 0_2, 0_3...
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {file_name} -> {new_name}")

print("Renaming complete!")

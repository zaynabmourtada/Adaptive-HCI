# This script labels all images used to train/test both the digit and letter recognizer models

import os

folder_path = "Deniz Collected Letters/Z"  

files = sorted(os.listdir(folder_path))  

# Loop through and rename files
for index, file_name in enumerate(files, start=1):
    file_ext = os.path.splitext(file_name)[1]  
    new_name = f"Z_{index}{file_ext}"  # Rename format: 0_1, 0_2, 0_3 or A_1, A_2, A_3
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {file_name} -> {new_name}")

print("Renaming complete!")

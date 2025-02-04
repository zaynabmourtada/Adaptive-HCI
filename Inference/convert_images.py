import os
import cv2

# ENTER YOUR PATHS HERE
dataset_path = "C:/Users/zayna/OneDrive/Documents/University/Senior Design/adaptive_code/Adaptive-HCI/Inference/10000 DIDA/9"   # Folder where raw images are stored
output_path = "C:/Users/zayna/OneDrive/Documents/University/Senior Design/adaptive_code/Adaptive-HCI/Inference/10000 DIDA/9_converted"     # Folder to save 28x28 images

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loop through all images in the dataset folder
for img_file in os.listdir(dataset_path):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Check if it's an image
        img_path = os.path.join(dataset_path, img_file)

        # Read image and convert to grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Resize to 28x28

        # Save the resized image in the output folder
        cv2.imwrite(os.path.join(output_path, img_file), img)

print(f" All images have been resized to 28x28 and saved in: {output_path}")

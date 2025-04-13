import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define paths relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")
input_image_path = os.path.join(script_dir, "input", "frame_0002.png")
output_image_path = os.path.join(script_dir, "output", "frame_0002.png")

# Load YOLO model and set device
model = YOLO(model_path).to(device)

# Read Input Image
image = cv2.imread(input_image_path)

# Convert IMG to Tensor
input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

# Run inference on the image (YOLO model processes an image directly)
results = model.predict(image, imgsz=416, device=device, verbose=False)
raw_results = model.model(input_tensor)
print(raw_results[0].shape)
print("0: ", raw_results[0][0, :, 0])
print()

box_data = results[0].boxes.data

for box in box_data:
    print(box)
    print()

# Define class label mapping and colors
class_labels = {0: "User_1", 1: "User_2", 2: "User_3"}
class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}

# Draw detections on a copy of the image
annotated_image = image.copy()
for box in box_data:
    x1, y1, x2, y2, conf, cls = box
    label = class_labels.get(int(cls), f"Class {int(cls)}")
    color = class_colors.get(int(cls), (0, 255, 0))
    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(annotated_image, f"{label} {conf:.2f}", (int(x1)-10, int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Ensure output directory exists and save the annotated image
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
cv2.imwrite(output_image_path, annotated_image)
print(f"\nAnnotated image saved to: {output_image_path}")

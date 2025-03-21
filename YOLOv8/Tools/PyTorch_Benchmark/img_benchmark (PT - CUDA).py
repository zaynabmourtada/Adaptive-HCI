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
# Automatically detect expected input size (default to 416 if not set)
expected_size = model.overrides.get("imgsz", 416)
print(f"Model expected image size: {expected_size}x{expected_size}")

# Create a dummy input tensor with shape [1, 3, expected_size, expected_size] and print raw output shape
dummy_input = torch.zeros((1, 3, expected_size, expected_size)).to(device)
with torch.no_grad():
    raw_output = model.model(dummy_input)
if isinstance(raw_output, (list, tuple)):
    for idx, out in enumerate(raw_output):
        if isinstance(out, torch.Tensor):
            print(f"Raw output tensor {idx} shape: {out.shape}")
        elif isinstance(out, list):
            print(f"Raw output tensor {idx} is a list with length: {len(out)}")
        else:
            print(f"Raw output tensor {idx} is of type {type(out)}")
else:
    print("Raw output tensor shape:", raw_output.shape)
# Note: For a YOLOv1*n model, if the second dimension is 7 then it implies 2 class scores, 
# if it's 8 it implies 3 class scores.

# Define class label mapping and colors
class_labels = {0: "User_1", 1: "User_2", 2: "User_3"}
class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}

# Load the input image
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Input image not found at {input_image_path}")
image = cv2.imread(input_image_path)
if image is None:
    raise ValueError("Failed to load the image.")
print(f"Input image loaded with dimensions: {image.shape[1]}x{image.shape[0]}")

# --------------------------
# Process the raw output tensor on the actual image:
# Preprocess the image (resize and convert to tensor)
input_image_resized = cv2.resize(image, (expected_size, expected_size))
# Note: cv2 loads images in BGR format. Adjust as needed.
input_tensor = torch.from_numpy(input_image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
with torch.no_grad():
    raw_out_actual = model.model(input_tensor)
if isinstance(raw_out_actual, (list, tuple)):
    raw_out_tensor = raw_out_actual[0]
else:
    raw_out_tensor = raw_out_actual
# Remove batch dimension: now shape is [7, 3549] (for example)
raw_out_tensor = raw_out_tensor.squeeze(0)
raw_out_np = raw_out_tensor.cpu().numpy()  # shape: [7, num_detections] (e.g., [7,3549])
# --------------------------

def print_best_detection_info_raw(raw_out_np, orig_width, orig_height):
    best_det_info = {}
    num_detections = raw_out_np.shape[1]
    for i in range(num_detections):
        # Each detection is a column vector of 7 values:
        # [x_center, y_center, width, height, obj_conf, score_class0, score_class1] 
        # (Note: For a 7-value tensor, there are 2 class scores; adjust if you have more.)
        detection = raw_out_np[:, i]
        x_center, y_center, width, height, obj_conf = detection[:5]
        class_scores = detection[5:]
        # Compute final scores (element-wise multiplication)
        final_scores = obj_conf * class_scores
        cls = int(np.argmax(final_scores))
        final_conf = final_scores[cls]
        # Compute bounding box in pixel coordinates
        x_center_px = x_center * orig_width
        y_center_px = y_center * orig_height
        width_px = width * orig_width
        height_px = height * orig_height
        xmin = max(0, x_center_px - width_px / 2)
        ymin = max(0, y_center_px - height_px / 2)
        xmax = min(orig_width, x_center_px + width_px / 2)
        ymax = min(orig_height, y_center_px + height_px / 2)
        # Update best detection info per class (using raw detection index from 0 to num_detections-1)
        if cls not in best_det_info or final_conf > best_det_info[cls]['conf']:
            best_det_info[cls] = {
                'index': i,
                'raw': detection,
                'conf': final_conf,
                'bbox': (xmin, ymin, xmax, ymax)
            }
    # Print one line per class (if detection exists)
    for cls in sorted(class_labels.keys()):
        label = class_labels.get(cls, f"Class {cls}")
        if cls in best_det_info:
            info = best_det_info[cls]
            raw_str = np.array2string(info['raw'], separator=' ', threshold=np.inf, max_line_width=1000)
            xmin, ymin, xmax, ymax = info['bbox']
            print(f"Class: {label}, Detection Index: {info['index']}, Scaled Confidence: {info['conf']:.3f}\nRaw Tensor: {raw_str}\nBounding Box: ({xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f})")
        else:
            print(f"No detection for {label}")

# Print best detection info from the raw tensor (using the full range 0-3549)
print_best_detection_info_raw(raw_out_np, image.shape[1], image.shape[0])

# --------------------------
# Now run inference using the high-level predict API and draw detections

# Run inference on the image (YOLO model processes an image directly)
results = model.predict(image, imgsz=expected_size, device=device, verbose=False)
result = results[0]
# Get detections: each row is [xmin, ymin, xmax, ymax, confidence, class]
detections = result.boxes.data.cpu().numpy() if result.boxes.data is not None else np.array([])
print(f"\nNumber of detections: {len(detections)}")

# (Optional) Print each post-processed detection for debugging
for i, det in enumerate(detections):
    xmin, ymin, xmax, ymax, confidence, cls = det
    label = class_labels.get(int(cls), f"Class {int(cls)}")
    print(f"Detection {i}: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}], confidence: {confidence:.2f}, class: {int(cls)} ({label})")

# Draw detections on a copy of the image
annotated_image = image.copy()
for det in detections:
    xmin, ymin, xmax, ymax, confidence, cls = det
    label = class_labels.get(int(cls), f"Class {int(cls)}")
    color = class_colors.get(int(cls), (0, 255, 0))
    cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    cv2.putText(annotated_image, f"{label} {confidence:.2f}", (int(xmin), int(ymin)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Ensure output directory exists and save the annotated image
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
cv2.imwrite(output_image_path, annotated_image)
print(f"\nAnnotated image saved to: {output_image_path}")

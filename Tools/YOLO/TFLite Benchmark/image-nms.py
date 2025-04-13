import cv2
import numpy as np
import tensorflow as tf
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(script_dir, "best_float32-NMS-SIMPLIFY.tflite")
input_image_path = os.path.join(script_dir, "input", "frame_0002.png")
output_image_path = os.path.join(script_dir, "output", "frame_0002-NMS-SIMPLIFY.png")

# Load the TFLite model and allocate tensors
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image (assuming image is already 416x416)
print("Loading input image...")
image = cv2.imread(input_image_path)
if image is None:
    raise ValueError(f"Failed to load image from {input_image_path}")

# The expected image size is 416x416 in this example.
img_width, img_height = 416, 416

# Expand dimensions to add batch size (assumes model expects a batch dimension)
input_data = np.expand_dims(image, axis=0).astype(np.float32)

# Set the tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Retrieve the raw output tensor (assumed shape: (1, 300, 6))
raw_output = interpreter.get_tensor(output_details[0]['index'])
print("Tensor shape:", raw_output.shape)

# Assume output tensor shape is (1, 300, 6)
# Each detection: [x1, y1, x2, y2, conf, label]
detections = raw_output[0]  # shape (300, 6)

# Dictionary to hold the top detection for each class
top_detections = {}

# Find the top detection for each class (0, 1, 2)
for class_id in [0, 1, 2]:
    # Filter detections for the current class
    mask = detections[:, 5] == class_id
    if np.any(mask):
        filtered_dets = detections[mask]
        # Find the index of the detection with highest confidence (column index 4)
        top_index = np.argmax(filtered_dets[:, 4])
        top_det = filtered_dets[top_index]
        top_detections[class_id] = top_det
        print(f"Top detection for class {class_id}: {top_det}")
    else:
        print(f"No detections for class {class_id}")

# Create a copy of the original image for annotation
annotated_image = image.copy()

# Define colors for each class (BGR format for OpenCV)
class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
class_labels = {0: "User_1", 1: "User_2", 2: "User_3"}

# Draw bounding boxes and labels on the image
for class_id, det in top_detections.items():
    # The detection format: [x1, y1, x2, y2, conf, label] in normalized coordinates
    x1, y1, x2, y2, conf, label = det
    # Scale coordinates to the image size (416x416)
    xmin = int(x1 * img_width)
    ymin = int(y1 * img_height)
    xmax = int(x2 * img_width)
    ymax = int(y2 * img_height)
    
    # Draw a rectangle
    color = class_colors.get(class_id, (0, 255, 0))
    cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Prepare label text with confidence score
    label_text = f"{class_labels.get(class_id, 'Class ' + str(class_id))}: {conf:.2f}"
    # Put the label text above the rectangle
    cv2.putText(annotated_image, label_text, (xmin, max(ymin - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# Save the annotated image
cv2.imwrite(output_image_path, annotated_image)
print(f"Annotated image saved to: {output_image_path}")

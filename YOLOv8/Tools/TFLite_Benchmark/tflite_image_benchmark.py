import cv2
import numpy as np
import tensorflow as tf
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(script_dir, "best_float32.tflite")  # Using the model without NMS
input_image_path = os.path.join(script_dir, "input", "frame_0002.png")
output_image_path = os.path.join(script_dir, "output", "frame_0002.png")

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

# Expected image size (for scaling)
img_width, img_height = 416, 416

# Expand dimensions to add batch size (assumes model expects a batch dimension)
input_data = np.expand_dims(image, axis=0).astype(np.float32)

# Set the tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Retrieve the raw output tensor
raw_output = interpreter.get_tensor(output_details[0]['index'])
print("Tensor shape:", raw_output.shape)
# raw_output shape is (1, 7, 3549)

# Transpose to get shape (3549, 7)
detections = raw_output[0].transpose()  # Now each row is a detection vector of 7 elements.
num_detections = detections.shape[0]
print("Number of detections:", num_detections)

# Dictionary to store the top detection per class (with detection index)
top_detections = {}

print("0: ", detections[0])

# For each class (0, 1, 2), find the detection with the highest score.
# Here, detection vector: [x, y, width, height, score_class0, score_class1, score_class2]
for class_id in [0, 1, 2]:
    top_score = -1.0
    top_det = None
    top_index = -1
    for i in range(num_detections):
        det = detections[i]
        # Score for current class is at index 4+class_id
        score = det[4 + class_id]
        if score > top_score:
            top_score = score
            top_det = det
            top_index = i
    if top_det is not None:
        top_detections[class_id] = (top_det, top_index)
        print(f"\nTop detection for class {class_id} is at index {top_index}:")
        print(top_det)
    else:
        print(f"\nNo detections for class {class_id}")

# Create a copy of the image for annotation
annotated_image = image.copy()

# Define colors and labels for each class (BGR for OpenCV)
class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
class_labels = {0: "User_1", 1: "User_2", 2: "User_3"}

# Draw bounding boxes and labels for the top detections
for class_id, (det, det_index) in top_detections.items():
    # Unpack detection: [x, y, width, height, score0, score1, score2]
    x, y, width, height = det[0], det[1], det[2], det[3]
    # The model outputs normalized center coordinates along with width and height.
    # Convert these to corner coordinates:
    xmin = int((x - width / 2) * img_width)
    ymin = int((y - height / 2) * img_height)
    xmax = int((x + width / 2) * img_width)
    ymax = int((y + height / 2) * img_height)
    
    # Confidence for the detected class
    conf = det[4 + class_id]
    
    # Draw bounding box
    color = class_colors.get(class_id, (0, 255, 0))
    cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Prepare label text including detection index and confidence
    label_text = f"{class_labels.get(class_id, 'Class ' + str(class_id))} idx:{det_index} conf:{conf:.2f}"
    cv2.putText(annotated_image, label_text, (xmin, max(ymin - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# Save the annotated image
cv2.imwrite(output_image_path, annotated_image)
print("\nAnnotated image saved to:", output_image_path)

import os
import torch
import cv2
import numpy as np

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Path to the TorchScript model
model_path = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights\best.torchscript"

# Ensure the model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

# Load the TorchScript model
print(f"📥 Loading TorchScript model from {model_path}...")
model = torch.jit.load(model_path, map_location=device)
model.eval()  # Set model to evaluation mode
print(f"✅ Model successfully loaded!")

# Print the computation graph
#print(f"🔍 Model computation graph:")
#print(model.graph)

# Path to test image
test_image_path = "./4_validation_images/input/frame_0128.jpg"

# Ensure the image exists
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"❌ Test image not found at: {test_image_path}")

# Load and preprocess the image
image = cv2.imread(test_image_path)
if image is None:
    raise ValueError("❌ Error: Unable to load the test image.")

# Convert image to RGB (OpenCV loads in BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image to model input size (960x960)
image_resized = cv2.resize(image_rgb, (960, 960))

# Normalize pixel values (0-255 → 0-1) and convert to tensor
image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0  # Shape: [3, 960, 960]
image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim: [1, 3, 960, 960]

# Perform inference
print(f"🚀 Running inference on test image...")
with torch.no_grad():
    output = model(image_tensor)

# Log full tensor shape
#print(f"📏 Model Output Tensor Shape: {output.shape}")

# Ensure correct format
if isinstance(output, torch.Tensor):
    output_array = output.cpu().numpy()
    print(f"🔢 First 10 output values: {output_array.flatten()[:10].tolist()}")

    # Extract bounding boxes
    num_detections = output.shape[2]  # Should be the number of detections
    print("📌 Number of detections:", num_detections)

for i in range(output.shape[2]):
    x_center, y_center, width, height, confidence = output[0, :, i].tolist()
    if confidence > 0.8:
        print(f"DETECTION {i}: x_center={x_center:.8f}, y_center={y_center:.8f}, width={width:.8f}, height={height:.8f} confidence={confidence:.8f}")
else:
    print(f"⚠️ Unexpected output format: {type(output)}")

print("🎯 Model test completed successfully!")

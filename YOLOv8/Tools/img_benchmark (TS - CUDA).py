import os
import torch
import cv2
import numpy as np

def set_device():
    """Set the device (CPU or CUDA)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    return device

def load_model(model_path, device):
    """Load the TorchScript model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
    print(f"📥 Loading TorchScript model from {model_path}...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()  # Set model to evaluation mode
    print(f"✅ Model successfully loaded!")
    return model

def load_image(image_path):
    """Load and validate the input image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Test image not found at: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Error: Unable to load the test image.")
    return image

def preprocess_image(image):
    """Preprocess the image for inference."""
    # Convert image to RGB (OpenCV loads in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize pixel values (0-255 → 0-1) and convert to tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0  # Shape: [3, H, W]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dim: [1, 3, H, W]
    return image_tensor

def run_inference(model, image_tensor, device):
    """Run inference on the input tensor."""
    print(f"🚀 Running inference on test image...")
    with torch.no_grad():
        output = model(image_tensor.to(device))
    return output

def find_best_detection(output):
    """Find the detection with the highest confidence."""
    max_confidence = -1
    best_detection = None
    for i in range(output.shape[2]):
        x_center, y_center, width, height, confidence = output[0, :, i].tolist()
        if confidence > max_confidence:
            max_confidence = confidence
            best_detection = (x_center, y_center, width, height, confidence)
    return best_detection

def draw_bounding_box(image, detection, output_image_path):
    """Draw the bounding box and confidence on the image."""
    x_center, y_center, width, height, confidence = detection
    # Calculate bounding box coordinates
    x1 = int(x_center - (width / 2))
    y1 = int(y_center - (height / 2))
    x2 = int(x_center + (width / 2))
    y2 = int(y_center + (height / 2))
    # Draw the bounding box
    box_color = (0, 255, 0)  # Green color (BGR format)
    box_thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)
    # Write the confidence value
    text = f"Confidence: {confidence:.4f}"
    text_position = (x1, y1 - 10)  # Position the text above the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (0, 255, 0)  # Green color (BGR format)
    text_thickness = 2
    cv2.putText(image, text, text_position, font, font_scale, text_color, text_thickness)
    # Save the output image
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, image)
    print(f"💾 Output image saved to: {output_image_path}")

def main():
    # Set device
    device = set_device()

    # Paths
    model_path = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights\best.torchscript"
    test_image_path = "./4_validation_images/input/frame_0128.jpg"
    output_image_path = "./4_validation_images/output/frame_0128_TS.jpg"

    # Load model
    model = load_model(model_path, device)

    # Load and preprocess image
    image = load_image(test_image_path)
    image_tensor = preprocess_image(image)

    # Run inference
    output = run_inference(model, image_tensor, device)

    # Log output tensor shape
    print(f"📏 Model Output Tensor Shape: {output.shape}")

    # Ensure correct format
    if isinstance(output, torch.Tensor):
        output_array = output.cpu().numpy()
        print(f"🔢 First 10 output values: {output_array.flatten()[:10].tolist()}")
        print("📌 Number of detections:", output.shape[2])

    # Find the best detection
    best_detection = find_best_detection(output)
    if best_detection is not None:
        x_center, y_center, width, height, confidence = best_detection
        print(f"BEST DETECTION: x_center={x_center:.8f}, y_center={y_center:.8f}, width={width:.8f}, height={height:.8f}, confidence={confidence:.8f}")
        # Draw bounding box and save output image
        draw_bounding_box(image, best_detection, output_image_path)
    else:
        print("⚠️ No detections found.")

    print("🎯 Model test completed successfully!")

if __name__ == "__main__":
    main()
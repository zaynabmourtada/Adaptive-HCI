import os
import cv2
import numpy as np
import tensorflow.lite as tflite

def load_tflite_model(model_path):
    """Load the TFLite model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
    
    print(f"📥 Loading TFLite model from {model_path}...")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    print(f"✅ Model successfully loaded!")
    return interpreter

def load_image(image_path):
    """Load and validate the input image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Test image not found at: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Error: Unable to load the test image.")
    
    return image

def preprocess_image(image, input_shape):
    """Preprocess the image for inference."""
    # Convert image to RGB (OpenCV loads in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to match model input
    input_height, input_width, _ = input_shape
    image_resized = cv2.resize(image_rgb, (input_width, input_height))
    
    # Normalize pixel values (0-255 → 0-1)
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension: [1, H, W, 3]
    image_input = np.expand_dims(image_normalized, axis=0)
    
    return image_input

def run_inference(interpreter, image_input):
    """Run inference on the input image."""
    print(f"🚀 Running inference on test image...")
    
    # Get input & output tensor indices
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    # Set input tensor
    interpreter.set_tensor(input_index, image_input)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output = interpreter.get_tensor(output_index)
    
    return output

def find_best_detection(output, image_width, image_height):
    """Find the detection with the highest confidence and convert to pixel coordinates."""
    num_detections = output.shape[2]

    max_confidence = -1
    best_detection = None

    for i in range(num_detections):
        # Extract values (normalized)
        x_center, y_center, width, height, confidence = output[0, :, i].tolist()

        if confidence > max_confidence:
            max_confidence = confidence
            best_detection = (
                x_center * image_width,  # Scale x_center
                y_center * image_height,  # Scale y_center
                width * image_width,  # Scale width
                height * image_height,  # Scale height
                confidence
            )

    return best_detection


def draw_bounding_box(image, detection, output_image_path):
    """Draw the bounding box and confidence on the image (with scaled pixel coordinates)."""
    x_center, y_center, width, height, confidence = detection

    # Convert from center format (cx, cy, w, h) → (x1, y1, x2, y2)
    x1 = int(x_center - (width / 2))
    y1 = int(y_center - (height / 2))
    x2 = int(x_center + (width / 2))
    y2 = int(y_center + (height / 2))

    # Draw bounding box
    box_color = (0, 255, 0)  # Green color
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    # Draw confidence text
    text = f"Confidence: {confidence:.4f}"
    text_position = (x1, max(y1 - 10, 10))  # Prevent text from going above image
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the image
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, image)
    print(f"💾 Output image saved to: {output_image_path}")


def main():
    # Paths
    model_path = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights\best_saved_model\best_float32.tflite"
    test_image_path = "./4_validation_images/input/frame_0128.jpg"
    output_image_path = "./4_validation_images/output/frame_0128_TFLite.jpg"

    # Load model
    interpreter = load_tflite_model(model_path)

    # Get model input shape
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape'][1:4]  # Get (H, W, C)

    # Load and preprocess image
    image = load_image(test_image_path)
    image_input = preprocess_image(image, input_shape)

    # Run inference
    output = run_inference(interpreter, image_input)

    # Log output tensor shape
    print(f"📏 Model Output Tensor Shape: {output.shape}")

    # Ensure correct format
    print(f"🔢 First 10 output values: {output.flatten()[:10].tolist()}")
    print("📌 Number of detections:", output.shape[2])

    # Get original image dimensions
    image_height, image_width = image.shape[:2]

    # Find the best detection and scale it to pixel coordinates
    best_detection = find_best_detection(output, image_width, image_height)

    if best_detection is not None:
        print(f"BEST DETECTION (SCALED): {best_detection}")
        # Draw bounding box and save output image
        draw_bounding_box(image, best_detection, output_image_path)
    else:
        print("⚠️ No detections found.")

    print("🎯 Model test completed successfully!")

if __name__ == "__main__":
    main()

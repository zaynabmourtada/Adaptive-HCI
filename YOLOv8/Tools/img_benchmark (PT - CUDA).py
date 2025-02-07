import cv2
import os
import time  # For timing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from ultralytics import YOLO  # Ensure YOLOv8 library is installed
import torch

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Optimize OpenCV performance
cv2.setNumThreads(6)  # Set the number of threads based on your CPU cores
cv2.setUseOptimized(True)  # Enable optimization

# Paths to model, input images, and output folder
model_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Models\\3\\weights\\best.pt"  # Path to your YOLO model
input_dir = "./4_validation_images/input"  # Directory containing input images
output_dir = "./4_validation_images/output"  # Directory to save processed images

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model with the specified device
model = YOLO(model_path).to(device)

# Automatically detect expected input size
expected_size = model.overrides.get("imgsz", 640)  # Get imgsz if available, else default to 640
print(f"Model expected image size: {expected_size}x{expected_size}")

# Function to process a batch of images
def process_batch(images, image_paths, output_dir, progress_bar):
    try:
        # Store original image sizes (width, height)
        original_sizes = [(img.shape[1], img.shape[0]) for img in images]

        # Resize images to model's expected size
        resized_images = [cv2.resize(img, (expected_size, expected_size)) for img in images]

        # Run batch inference
        results = model.predict(resized_images, imgsz=expected_size, device=device, verbose=False)

        for i, (result, image_path) in enumerate(zip(results, image_paths)):
            detections = result.boxes.data.cpu().numpy() if result.boxes.data is not None else []
            original_image = images[i].copy()  # Keep original image for drawing
            
            # Get the original image size
            orig_w, orig_h = original_sizes[i]
            
            # Calculate scaling factors to map bounding boxes back to the original image size
            scale_x = orig_w / expected_size
            scale_y = orig_h / expected_size

            # Draw detections on the original image
            for det in detections:
                xmin, ymin, xmax, ymax, confidence, cls = det
                label = f"Class {int(cls)}"

                # Scale bounding boxes back to original image size
                xmin, xmax = int(xmin * scale_x), int(xmax * scale_x)
                ymin, ymax = int(ymin * scale_y), int(ymax * scale_y)

                # Draw bounding boxes on the original-sized image
                cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(original_image, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the processed image (original size with bounding boxes)
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_PT.jpg")
            cv2.imwrite(output_path, original_image)
            progress_bar.update(1)  # Update the progress bar
            print(f"Saved processed image to {output_path}")

    except Exception as e:
        print(f"Error processing batch: {e}")

# Function to process all images in batches
def process_images_in_batches(input_dir, output_dir, batch_size=8):
    images = []
    image_paths = []

    # Initialize progress bar
    total_images = len(os.listdir(input_dir))
    with tqdm(total=total_images, desc="Processing Images", unit="image") as progress_bar:
        start_time = None  # Timer starts when the first image is processed
        for image_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, image_name)
            image = cv2.imread(input_path)

            if image is None:
                print(f"Error: Unable to read image {input_path}")
                progress_bar.update(1)  # Skip the image but update progress
                continue

            # Start the timer when the first valid image is processed
            if start_time is None:
                start_time = time.time()

            images.append(image)
            image_paths.append(input_path)

            # Process a batch when it's full
            if len(images) == batch_size:
                process_batch(images, image_paths, output_dir, progress_bar)
                images.clear()
                image_paths.clear()

        # Process any remaining images
        if images:
            process_batch(images, image_paths, output_dir, progress_bar)

        # End the timer after the final image is processed
        if start_time is not None:
            end_time = time.time()
            print(f"Total processing time: {end_time - start_time:.2f} seconds")

# Main function with threading
def process_images_threaded(input_dir, output_dir, max_workers=4, batch_size=8):
    print("Starting threaded image processing...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.submit(process_images_in_batches, input_dir, output_dir, batch_size)

if __name__ == "__main__":
    process_images_threaded(input_dir, output_dir, max_workers=6, batch_size=8)
    print("Image processing complete.")

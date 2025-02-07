import cv2
import torch
from tqdm import tqdm
import time

# Force the device to CPU to simulate phone-like environment
device = "cpu"
print(f"Using device: {device}")

# Paths to model, input video, and output video
model_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Models\\yolo_v2\\weights\\best_optimized.torchscript"  # Path to optimized TorchScript model
input_video_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Tools\\0_video\\padded_video.mp4"  # Input video path
output_video_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Tools\\0_video\\proc_video.mp4"  # Output video path

# Load TorchScript model on CPU
model = torch.jit.load(model_path).to(device)
model.eval()  # Set the model to evaluation mode

# Function to process a single frame
def process_frame(frame):
    # Preprocess the frame: Normalize and convert to tensor
    input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Parse outputs (adjust based on the model's expected output structure)
    detections = []
    if len(output) > 0 and isinstance(output[0], torch.Tensor):
        detections = output[0].cpu().numpy()

    # Draw detections on the frame
    for det in detections:
        if len(det) >= 6:  # Ensure there are at least 6 values to unpack
            xmin, ymin, xmax, ymax, confidence, cls = det[:6]  # Unpack only the first 6 values
            label = f"Class {int(cls)}"
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame



# Function to process the video
def process_video(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Progress bar
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        start_time = time.time()  # Start timer
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = process_frame(frame)

            # Write the processed frame to the output video
            out.write(processed_frame)

            # Update progress bar
            pbar.update(1)

        # Release resources
        cap.release()
        out.release()
        end_time = time.time()  # End timer
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    process_video(input_video_path, output_video_path)
    print("Video processing complete.")

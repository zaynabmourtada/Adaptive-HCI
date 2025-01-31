import cv2
import os
from tqdm import tqdm
from ultralytics import YOLO  # Ensure YOLOv8 library is installed
import torch
import time

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths to model, input video, and output video
model_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Models\\yolo_v2\\weights\\best.pt"  # Path to your YOLO model
input_video_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Tools\\0_video\\padded_video.mp4"  # Input video path
output_video_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Tools\\0_video\\proc_video.mp4"  # Output video path

# Load YOLO model with the specified device
model = YOLO(model_path).to(device)

# Function to process a single frame
def process_frame(frame):
    # Run inference on the frame
    results = model.predict(frame, imgsz=960, device=device, verbose=False)

    # Draw detections on the frame
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []
    for det in detections:
        xmin, ymin, xmax, ymax, confidence, cls = det
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

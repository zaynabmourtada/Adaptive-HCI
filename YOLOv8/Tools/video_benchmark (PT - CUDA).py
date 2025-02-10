import cv2
import os
import torch
import time
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths to model, input video, and output video
model_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Models\\8\\weights\\best.pt"  # Path to your YOLO model
input_video_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Tools\\0_video\\yoloV6.mp4"  # Input video path
output_video_path = "F:\\GitHub\\Adaptive-HCI\\YOLOv8\\Tools\\0_video\\proc_video.mp4"  # Output video path

# Load YOLO model with the specified device
model = YOLO(model_path).to(device)

# Automatically detect expected input size
expected_size = model.overrides.get("imgsz", 416)  # Get imgsz if available, else default to 640
print(f"Model expected image size: {expected_size}x{expected_size}")

# Function to process a batch of frames
def process_batch(frames, original_sizes):
    # Resize all frames for YOLO in one step
    #resized_frames = [cv2.resize(frame, (expected_size, expected_size)) for frame in frames]

    # Run batch inference (YOLO processes multiple images at once)
    results = model.predict(frames, imgsz=expected_size, device=device, verbose=False)

    processed_frames = []
    for i, (result, frame) in enumerate(zip(results, frames)):
        detections = result.boxes.data.cpu().numpy() if result.boxes.data is not None else []
        orig_w, orig_h = original_sizes[i]
        scale_x = orig_w / expected_size
        scale_y = orig_h / expected_size

        # Draw bounding boxes on the original frame
        for det in detections:
            xmin, ymin, xmax, ymax, confidence, cls = det
            label = f"Class {int(cls)}"

            # Scale bounding boxes back to original video size
            xmin, xmax = int(xmin * scale_x), int(xmax * scale_x)
            ymin, ymax = int(ymin * scale_y), int(ymax * scale_y)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        processed_frames.append(frame)

    return processed_frames

# Function to process video using batch inference
def process_video(input_path, output_path, batch_size=8):
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

    # Initialize progress bar
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        frames = []
        original_sizes = []
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            original_sizes.append((width, height))

            # Process batch when full
            if len(frames) == batch_size:
                processed_frames = process_batch(frames, original_sizes)
                for pf in processed_frames:
                    out.write(pf)

                frames.clear()
                original_sizes.clear()
                pbar.update(batch_size)

        # Process any remaining frames
        if frames:
            processed_frames = process_batch(frames, original_sizes)
            for pf in processed_frames:
                out.write(pf)
            pbar.update(len(frames))

        # Release resources
        cap.release()
        out.release()
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    process_video(input_video_path, output_video_path, batch_size=16)
    print("Video processing complete.")

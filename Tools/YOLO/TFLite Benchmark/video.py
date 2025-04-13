import cv2
import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Define script directory and paths
script_dir = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(script_dir, "best_float32.tflite")
input_folder = os.path.join(script_dir, "input")
output_folder = os.path.join(script_dir, "output")

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get expected image size from the model's input shape (assumed to be square)
input_shape = input_details[0]['shape']
expected_size = input_shape[1]
print(f"Model expected image size: {expected_size}x{expected_size}")

# Define class label mapping and colors for each class
class_labels = {
    0: "User_1",
    1: "User_2",
    2: "User_3"
}

class_colors = {
    0: (0, 255, 0),   # Green for User_1
    1: (255, 0, 0),   # Blue for User_2
    2: (0, 0, 255)    # Red for User_3
}

# -------------------------------
# Helper Functions
# -------------------------------

def iou(box1, box2):
    """Compute Intersection over Union (IoU) of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def non_max_suppression(detections, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter detections.
    Each detection is assumed to be [xmin, ymin, xmax, ymax, confidence, class].
    """
    if len(detections) == 0:
        return []
    nms_dets = []
    # Process detections per class
    for cls in np.unique(detections[:, 5]):
        cls_mask = detections[:, 5] == cls
        cls_dets = detections[cls_mask]
        # Sort by confidence (highest first)
        order = cls_dets[:, 4].argsort()[::-1]
        cls_dets = cls_dets[order]
        while len(cls_dets) > 0:
            best = cls_dets[0]
            nms_dets.append(best)
            if len(cls_dets) == 1:
                break
            rest = cls_dets[1:]
            ious = np.array([iou(best[:4], det[:4]) for det in rest])
            cls_dets = rest[ious < iou_threshold]
    return np.array(nms_dets) if nms_dets else np.array([])

def process_yolov8_outputs(output_data, orig_width, orig_height, conf_threshold=0.25):
    """
    Process YOLOv8 model outputs. The format is typically:
    [batch, 7, num_boxes] where:
    - values 0-3: center_x, center_y, width, height (normalized)
    - value 4: confidence
    - values 5-6: class confidences
    
    Returns: List of [xmin, ymin, xmax, ymax, confidence, class_id] detections
    """
    # Extract the shape information
    num_detections = output_data.shape[2]
    
    detections = []
    for i in range(num_detections):
        detection = output_data[0, :, i]
        # Extract values
        x_center, y_center, width, height, obj_conf = detection[:5]
        class_scores = detection[5:]
        
        # Multiply objectness with each class score to get final scores
        obj_conf = 1 / (1 + np.exp(-obj_conf))
        class_scores = 1 / (1 + np.exp(-class_scores))
        final_scores = obj_conf * class_scores
        class_id = np.argmax(final_scores)
        final_conf = final_scores[class_id]
        #print(f"Detection {i}: Objectness={obj_conf:.3f}, Class Score={class_scores[class_id]:.3f}, Final Conf={final_conf:.3f}")        
        # Skip detection if the final confidence is below threshold
        if final_conf < conf_threshold:
            continue

        # Convert normalized coordinates to pixel coordinates
        x_center_px = x_center * orig_width
        y_center_px = y_center * orig_height
        width_px = width * orig_width
        height_px = height * orig_height

        xmin = max(0, x_center_px - width_px / 2)
        ymin = max(0, y_center_px - height_px / 2)
        xmax = min(orig_width, x_center_px + width_px / 2)
        ymax = min(orig_height, y_center_px + height_px / 2)

        detections.append([xmin, ymin, xmax, ymax, final_conf, class_id])

    return np.array(detections) if detections else np.array([])

def run_inference_on_frame(frame, conf_threshold=0.25, nms_threshold=0.45):
    """
    Run object detection inference on a single frame.
    """
    # Get original frame dimensions
    orig_height, orig_width = frame.shape[:2]
    
    # Preprocess the frame
    # 1. Resize to the model's expected input size
    resized = cv2.resize(frame, (expected_size, expected_size))
    
    # 2. Convert from BGR to RGB (TensorFlow models expect RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize pixel values to [0,1]
    input_data = rgb.astype(np.float32) / 255.0
    
    # 4. Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process YOLOv8 format output
    detections = process_yolov8_outputs(
        output_data, 
        orig_width, 
        orig_height, 
        conf_threshold
    )
    
    # Apply NMS to filter overlapping boxes
    if len(detections) > 0:
        filtered_detections = non_max_suppression(detections, iou_threshold=nms_threshold)
    else:
        filtered_detections = np.array([])
    
    return filtered_detections

def process_batch(frames, conf_threshold=0.25, nms_threshold=0.45):
    """
    Process a batch of frames through the object detection model.
    """
    processed_frames = []
    for i, frame in enumerate(frames):
        # Run inference on the frame
        detections = run_inference_on_frame(frame, conf_threshold, nms_threshold)
        
        # Draw bounding boxes for all detections
        for det in detections:
            xmin, ymin, xmax, ymax, confidence, class_id = det
            
            # Convert to integers for drawing
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            
            # Get class label and color
            class_id = int(class_id)
            label = class_labels.get(class_id, f"Class {class_id}")
            color = class_colors.get(class_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Add label with confidence
            text = f"{label} {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for text
            cv2.rectangle(frame, 
                         (xmin, ymin - text_size[1] - 5), 
                         (xmin + text_size[0], ymin), 
                         color, -1)
            
            # Draw text
            cv2.putText(frame, text, (xmin, ymin - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        processed_frames.append(frame)
        
        # Print detection count periodically
        if i == 0 or i % 100 == 0:
            print(f"Frame {i}: {len(detections)} detections")
            if len(detections) > 0:
                # Print sample detection
                sample = detections[0]
                x1, y1, x2, y2, conf, cls = sample
                print(f"  Sample: Class={int(cls)}, Conf={conf:.2f}, Box=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    
    return processed_frames

def process_video(input_path, output_path, batch_size=8, conf_threshold=0.25, nms_threshold=0.45):
    """
    Process an entire video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        frames = []
        start_time = time.time()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_idx += 1
            
            # Process a batch of frames when we've collected enough or reached the end
            if len(frames) == batch_size or frame_idx == frame_count:
                processed_frames = process_batch(frames, conf_threshold, nms_threshold)
                
                # Write processed frames to output video
                for pf in processed_frames:
                    out.write(pf)
                
                # Update progress bar
                pbar.update(len(frames))
                
                # Clear the frames list for the next batch
                frames.clear()
        
        # Clean up
        cap.release()
        out.release()
        
        # Print processing statistics
        total_time = time.time() - start_time
        fps_processing = frame_count / total_time
        
        print(f"Processing complete:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average speed: {fps_processing:.2f} fps")
        print(f"  Output saved to: {output_path}")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    # Set detection parameters
    confidence_threshold = 0.05  # Set based on your needs
    nms_threshold = 0.45        # Set based on your needs
    
    print(f"Using confidence threshold: {confidence_threshold}")
    print(f"Using NMS threshold: {nms_threshold}")
    
    # Process each video in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            in_video = os.path.join(input_folder, filename)
            out_video = os.path.join(output_folder, filename)
            
            print(f"\nStarting processing for: {filename}")
            process_video(
                in_video, 
                out_video, 
                batch_size=16,
                conf_threshold=confidence_threshold,
                nms_threshold=nms_threshold
            )
    
    print("All videos processed.")
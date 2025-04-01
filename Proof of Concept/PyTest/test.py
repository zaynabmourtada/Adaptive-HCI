import os
import cv2
import numpy as np
from tqdm import tqdm

# ----------------- CONFIGURABLE PARAMETERS -----------------
THRESHOLD_VALUE = 10                   # Intensity threshold for binary thresholding (0-255)
MERGE_DISTANCE = 50                    # Maximum horizontal gap (in pixels) for merging bounding boxes
MERGE_BAR_WIDTH_RATIO_THRESHOLD = 0.8  # Minimum ratio of average bar widths for merging (e.g. 0.8 means similar enough)
MIN_BOX_AREA = 200                     # Minimum area (in pixels^2) for a bounding box to be considered valid
MIN_HEIGHT_WIDTH_RATIO = 0.5           # Minimum allowed height/width ratio for a bounding box
NUM_SAMPLES = 33                       # Number of horizontal rows sampled in each ROI for bar analysis
VALID_EXTS = ('.mp4', '.avi', '.mov', '.mkv')  # Video file extensions to process
INPUT_FOLDER = "input"                 # Input folder (relative to script directory)
OUTPUT_FOLDER = "output"               # Output folder (relative to script directory)
# -----------------------------------------------------------

def boxes_are_close_horizontal(box1, box2, merge_distance):
    """
    Determine if two boxes (x1, y1, x2, y2) are close enough horizontally to merge.
    Conditions:
      - Horizontal gap between them <= merge_distance.
      - Vertical overlap is at least 50% of the height of the smaller box.
    """
    # Ensure box1 is to the left of box2.
    if box1[0] > box2[0]:
        box1, box2 = box2, box1
    gap = max(0, box2[0] - box1[2])
    if gap > merge_distance:
        return False
    y_top = max(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])
    overlap = y_bottom - y_top
    if overlap <= 0:
        return False
    height1 = box1[3] - box1[1]
    height2 = box2[3] - box2[1]
    if overlap < 0.5 * min(height1, height2):
        return False
    return True

def merge_two_boxes(box1, box2):
    """Merge two bounding boxes and return the merged box."""
    return (
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    )

def merge_boxes(boxes, merge_distance):
    """
    Repeatedly merge boxes that are horizontally close until no further merging is possible.
    'boxes' is a list of bounding boxes (x1, y1, x2, y2).
    """
    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip = set()
        for i in range(len(boxes)):
            if i in skip:
                continue
            current_box = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip:
                    continue
                if boxes_are_close_horizontal(current_box, boxes[j], merge_distance):
                    current_box = merge_two_boxes(current_box, boxes[j])
                    skip.add(j)
                    merged = True
            new_boxes.append(current_box)
        boxes = new_boxes
    return boxes

def get_bar_info(row):
    """
    For a given row (1D array), count every contiguous segment of white pixels (255)
    and return the average length of these segments.
    """
    lengths = []
    current_length = 0
    for pixel in row:
        if pixel == 255:
            current_length += 1
        else:
            if current_length > 0:
                lengths.append(current_length)
                current_length = 0
    if current_length > 0:
        lengths.append(current_length)
    count = len(lengths)
    avg_length = sum(lengths) / count if count > 0 else 0
    return avg_length

def compute_avg_bar_width(thresh, box, num_samples):
    """
    Compute the average bar width for a given bounding box from the thresholded image.
    'box' is (x1, y1, x2, y2). Sample num_samples horizontal rows from the ROI.
    """
    x1, y1, x2, y2 = box
    roi = thresh[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    roi_height, _ = roi.shape
    avg_lengths = []
    for k in range(num_samples):
        row_idx = int((k+1) * roi_height / (num_samples + 1))
        row = roi[min(row_idx, roi_height - 1), :]
        avg_lengths.append(get_bar_info(row))
    overall_avg = sum(avg_lengths) / len(avg_lengths) if avg_lengths else 0
    return overall_avg

def merge_box_info(box_info, merge_distance, num_samples, thresh):
    """
    Given a list of tuples (box, avg_bar_width), repeatedly merge boxes that are horizontally
    close and have similar average bar widths. Two boxes are merged only if:
      - They are horizontally close (using boxes_are_close_horizontal).
      - The ratio of their avg_bar_width values is at least MERGE_BAR_WIDTH_RATIO_THRESHOLD.
    Returns a list of merged tuples (merged_box, merged_avg_bar_width).
    """
    merged = True
    while merged:
        merged = False
        new_box_info = []
        skip = set()
        for i in range(len(box_info)):
            if i in skip:
                continue
            current_box, current_avg = box_info[i]
            for j in range(i+1, len(box_info)):
                if j in skip:
                    continue
                other_box, other_avg = box_info[j]
                if boxes_are_close_horizontal(current_box, other_box, merge_distance):
                    # Check average bar width similarity:
                    if current_avg == 0 or other_avg == 0:
                        ratio = 1
                    else:
                        ratio = min(current_avg, other_avg) / max(current_avg, other_avg)
                    if ratio >= MERGE_BAR_WIDTH_RATIO_THRESHOLD:
                        merged_box = merge_two_boxes(current_box, other_box)
                        # Update average bar width; here we simply average the two values.
                        new_avg = (current_avg + other_avg) / 2
                        current_box, current_avg = merged_box, new_avg
                        skip.add(j)
                        merged = True
            new_box_info.append((current_box, current_avg))
        box_info = new_box_info
    return box_info

def process_video(video_path, output_path, threshold_value, merge_distance, min_box_area, num_samples):
    """
    Process a video by:
      1. Converting each frame to grayscale and thresholding it.
      2. Finding and merging contours into ROIs.
      3. For each ROI, computing the average bar width.
      4. Merging ROIs that are horizontally close and have similar average bar widths.
      5. Filtering out ROIs with an area below MIN_BOX_AREA or with height/width ratio below MIN_HEIGHT_WIDTH_RATIO.
      6. Sorting ROIs by average bar width and labeling the top three:
            - User_1 (largest average bar width) in green,
            - User_2 in red,
            - User_3 in blue.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Threshold the frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        processed_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Find contours and compute initial bounding boxes.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [ (x, y, x+w, y+h) for (x, y, w, h) in [cv2.boundingRect(cnt) for cnt in contours] ]
        initial_merged_boxes = merge_boxes(boxes, merge_distance)

        # Create ROI info list: each element is (box, avg_bar_width)
        roi_info = []
        for box in initial_merged_boxes:
            avg_width = compute_avg_bar_width(thresh, box, num_samples)
            roi_info.append((box, avg_width))

        # Merge ROI info based on both proximity and similar average bar width.
        merged_roi_info = merge_box_info(roi_info, merge_distance, num_samples, thresh)
        merged_boxes = [box for (box, avg) in merged_roi_info]

        # Filter boxes by area and by height/width ratio.
        final_boxes = []
        for (x1, y1, x2, y2) in merged_boxes:
            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height
            ratio = box_height / box_width if box_width != 0 else 0
            if area >= min_box_area and ratio >= MIN_HEIGHT_WIDTH_RATIO:
                final_boxes.append((x1, y1, x2, y2))

        # For each final ROI, compute overall average bar width.
        roi_data = []
        for (x1, y1, x2, y2) in final_boxes:
            roi = thresh[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_height, _ = roi.shape
            avg_lengths = []
            for k in range(num_samples):
                row_idx = int((k+1) * roi_height / (num_samples + 1))
                row = roi[min(row_idx, roi_height - 1), :]
                avg_lengths.append(get_bar_info(row))
            overall_avg_width = sum(avg_lengths) / len(avg_lengths) if avg_lengths else 0
            roi_data.append((x1, y1, x2, y2, overall_avg_width))

        # Sort ROIs by overall average bar width (largest first).
        sorted_roi = sorted(roi_data, key=lambda x: x[4], reverse=True)

        # Define colors and labels for the top three ROIs.
        box_colors = {
            "User_1": (0, 255, 0),   # Green
            "User_2": (0, 0, 255),   # Red
            "User_3": (255, 0, 0)    # Blue
        }
        labels = ["User_1", "User_2", "User_3"]

        # Label the top three ROIs.
        for i, roi in enumerate(sorted_roi[:3]):
            x1, y1, x2, y2, overall_avg_width = roi
            label = labels[i]
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_colors[label], 2)
            # Annotate with: "User_x L:{overall_avg_width:.3f}" on the same row.
            cv2.putText(processed_frame, f"{label} L:{overall_avg_width:.3f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[label], 1, cv2.LINE_AA)

        out.write(processed_frame)
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames from {video_path}")

def test():
    print("TEST.PY")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, INPUT_FOLDER)
    output_folder = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(VALID_EXTS):
            video_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            print(f"Processing video: {file_name}")
            process_video(video_path, output_path, THRESHOLD_VALUE, MERGE_DISTANCE, MIN_BOX_AREA, NUM_SAMPLES)
            print(f"Saved processed video to: {output_path}")

if __name__ == "__main__":
    test()
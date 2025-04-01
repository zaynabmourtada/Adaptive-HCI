import os
import cv2
import numpy as np
from tqdm import tqdm

# ----------------- CONFIGURABLE PARAMETERS -----------------
THRESHOLD_VALUE = 10                   # Intensity threshold for binary thresholding (0-255)
MERGE_DISTANCE = 50                    # Maximum horizontal gap (in pixels) for merging ROIs
MERGE_BAR_WIDTH_RATIO_THRESHOLD = 0.2  # Minimum ratio between avg bar widths to allow merging
MIN_BOX_AREA = 200                     # Minimum area (in pixels^2) for an ROI to be valid
MIN_HEIGHT_WIDTH_RATIO = 0.5           # Minimum allowed height/width ratio for an ROI
NUM_SAMPLES = 33                       # Number of horizontal rows sampled in each ROI for bar analysis
VALID_EXTS = ('.mp4', '.avi', '.mov', '.mkv')  # Video file extensions to process
INPUT_FOLDER = "input"                 # Input folder (relative to script directory)
OUTPUT_FOLDER = "output"               # Output folder (relative to script directory)
# -----------------------------------------------------------

def boxes_are_close_horizontal(box1, box2, merge_distance):
    """
    Determine if two bounding boxes (x1, y1, x2, y2) are close enough horizontally to merge.
    Conditions:
      - Horizontal gap between them is <= merge_distance.
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
    """Merge two bounding boxes (x1,y1,x2,y2) and return the merged box."""
    return (
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    )

def compute_avg_bar_width(thresh, box, num_samples):
    """
    Compute the average bar width for the ROI defined by 'box' (x1, y1, x2, y2)
    in the thresholded image. Samples num_samples horizontal rows and returns the overall average.
    """
    x1, y1, x2, y2 = box
    roi = thresh[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    roi_height, _ = roi.shape
    avg_lengths = []
    for k in range(num_samples):
        row_idx = int((k + 1) * roi_height / (num_samples + 1))
        row = roi[min(row_idx, roi_height - 1), :]
        avg_lengths.append(get_bar_info(row))
    overall_avg = sum(avg_lengths) / len(avg_lengths) if avg_lengths else 0
    return overall_avg

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

class ROI:
    """Represents a region of interest (ROI) with bounding box and average bar width."""
    def __init__(self, x1, y1, x2, y2, avg_bar_width):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.avg_bar_width = avg_bar_width

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def area(self):
        return self.width() * self.height()

    def ratio(self):
        return self.height() / self.width() if self.width() != 0 else 0

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def merge(self, other, thresh, num_samples):
        """
        Merge another ROI into this one by combining bounding boxes and re-computing the avg_bar_width
        for the new merged region.
        """
        self.x1 = min(self.x1, other.x1)
        self.y1 = min(self.y1, other.y1)
        self.x2 = max(self.x2, other.x2)
        self.y2 = max(self.y2, other.y2)
        self.avg_bar_width = compute_avg_bar_width(thresh, self.box(), num_samples)

class VideoProcessor:
    """Processes a video to extract, merge, filter, and label ROIs based on barcode patterns."""
    def __init__(self, threshold_value, merge_distance, merge_ratio_threshold,
                 min_box_area, min_height_width_ratio, num_samples):
        self.threshold_value = threshold_value
        self.merge_distance = merge_distance
        self.merge_ratio_threshold = merge_ratio_threshold
        self.min_box_area = min_box_area
        self.min_height_width_ratio = min_height_width_ratio
        self.num_samples = num_samples

    def process_frame(self, frame):
        """Convert a frame to grayscale, threshold it, and create ROI objects from contours."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        processed_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Find contours and create ROI objects.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box = (x, y, x+w, y+h)
            avg_bar_width = compute_avg_bar_width(thresh, box, self.num_samples)
            rois.append(ROI(x, y, x+w, y+h, avg_bar_width))
        return processed_frame, thresh, rois

    def merge_rois(self, rois, thresh):
        """
        Merge ROI objects that are horizontally close and have similar avg_bar_width.
        Returns a new list of merged ROI objects.
        """
        merged = True
        while merged:
            merged = False
            new_rois = []
            skip = set()
            for i in range(len(rois)):
                if i in skip:
                    continue
                current_roi = rois[i]
                for j in range(i+1, len(rois)):
                    if j in skip:
                        continue
                    other_roi = rois[j]
                    if boxes_are_close_horizontal(current_roi.box(), other_roi.box(), self.merge_distance):
                        # Check similarity of avg_bar_width.
                        if current_roi.avg_bar_width == 0 or other_roi.avg_bar_width == 0:
                            ratio = 1
                        else:
                            ratio = min(current_roi.avg_bar_width, other_roi.avg_bar_width) / max(current_roi.avg_bar_width, other_roi.avg_bar_width)
                        if ratio >= self.merge_ratio_threshold:
                            current_roi.merge(other_roi, thresh, self.num_samples)
                            skip.add(j)
                            merged = True
                new_rois.append(current_roi)
            rois = new_rois
        return rois

    def filter_rois(self, rois):
        """Filter out ROIs that do not meet the minimum area or height/width ratio criteria."""
        final_rois = []
        for roi in rois:
            if roi.area() < self.min_box_area:
                continue
            if roi.ratio() < self.min_height_width_ratio:
                continue
            final_rois.append(roi)
        return final_rois

    def annotate_frame(self, processed_frame, rois):
        """Sort ROIs by avg_bar_width, then annotate the top three with user labels and colors."""
        rois.sort(key=lambda r: r.avg_bar_width, reverse=True)
        box_colors = {
            "User_1": (0, 255, 0),   # Green
            "User_2": (0, 0, 255),   # Red
            "User_3": (255, 0, 0)    # Blue
        }
        labels = ["User_1", "User_2", "User_3"]
        for i, roi in enumerate(rois[:3]):
            label = labels[i]
            cv2.rectangle(processed_frame, (roi.x1, roi.y1), (roi.x2, roi.y2), box_colors[label], 2)
            cv2.putText(processed_frame, f"{label} L:{roi.avg_bar_width:.3f}", (roi.x1, roi.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[label], 1, cv2.LINE_AA)
        return processed_frame

    def process_video(self, video_path, output_path):
        """Process the entire video file."""
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

            processed_frame, thresh, rois = self.process_frame(frame)
            rois = self.merge_rois(rois, thresh)
            rois = self.filter_rois(rois)
            annotated_frame = self.annotate_frame(processed_frame, rois)
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()
        print(f"Processed {frame_count} frames from {video_path}")

def obj_test():
    print("OBJ_TEST.PY")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, INPUT_FOLDER)
    output_folder = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(output_folder, exist_ok=True)

    processor = VideoProcessor(THRESHOLD_VALUE, MERGE_DISTANCE, MERGE_BAR_WIDTH_RATIO_THRESHOLD,
                               MIN_BOX_AREA, MIN_HEIGHT_WIDTH_RATIO, NUM_SAMPLES)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(VALID_EXTS):
            video_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            print(f"Processing video: {file_name}")
            processor.process_video(video_path, output_path)
            print(f"Saved processed video to: {output_path}")

if __name__ == "__main__":
    obj_test()

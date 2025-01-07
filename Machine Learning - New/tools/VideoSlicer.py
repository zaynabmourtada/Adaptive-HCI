import os
import cv2
import tkinter as tk
from tkinter import filedialog

def slice_video_to_images(video_path, output_dir, prefix="frame"):
    """
    Slices a video into individual frames saved as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the images will be saved.
        prefix (str): Prefix for naming the output image files.
    """
    
    # Create the output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0

    while True:
        # Read a new frame from the video
        ret, frame = cap.read()
        
        # If no frame is returned, we’ve reached the end of the video
        if not ret:
            break

        # Construct the file name for the current frame
        frame_file_name = f"{prefix}_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_file_name)
        
        # Save the frame as a .jpg file
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Total frames extracted: {frame_count}")
    print("Slicing completed!")

def main():
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    # Prompt user to select a video file
    print("Please select the video file to slice...")
    video_file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"), ("All Files", "*.*")]
    )

    # If no file selected, exit
    if not video_file_path:
        print("No video file selected. Exiting...")
        return

    # Prompt user to select an output folder
    print("Please select the folder where images will be saved...")
    output_folder_path = filedialog.askdirectory(
        title="Select Output Folder"
    )

    # If no folder selected, exit
    if not output_folder_path:
        print("No output folder selected. Exiting...")
        return

    # (Optional) Prompt user for prefix input or just use a default
    prefix = "my_video_frame"  # You can hardcode or prompt user if needed
    
    # Slice the video into images
    slice_video_to_images(video_file_path, output_folder_path, prefix)

if __name__ == "__main__":
    main()

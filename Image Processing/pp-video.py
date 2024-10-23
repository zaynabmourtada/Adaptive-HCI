# Soham Naik - Senior Design - Adaptive HCI - Pre-Processing & Video Pathing DEMO - 10/17/2024

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


def ask_user_for_video_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select the input video file",
                                           filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    return file_path


def read_video(video_path):  # Read the video from the given path
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Video not found at {video_path}")
        exit()
    return video


def convert_to_grayscale(image):  # Convert the input frame to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_brightness(image, cutoff=250, dark_factor=0.3, bright_factor=2.0):  # Enhance brightness of the image
    enhanced_image = image.copy()  # Create a copy of the image to enhance brightness
    bright_pixels = image > cutoff  # Amplify bright areas (pixels above the cutoff)
    enhanced_image[bright_pixels] = np.clip(image[bright_pixels] * bright_factor, 0, 255)
    dark_pixels = image <= cutoff  # Darken dark areas (pixels below or equal to the cutoff)
    enhanced_image[dark_pixels] = np.clip(image[dark_pixels] * dark_factor, 0, 255)
    return enhanced_image.astype(np.uint8)


def overlay_edges(image, edge_points, dot_radius=3, dot_color=(0, 0, 255)):  # Overlay red edges on the image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale image to BGR
    for (y, x) in edge_points:
        cv2.circle(output_image, (x, y), dot_radius, dot_color, -1)  # Draw red dots at edge points
    return output_image


def overlay_center(output_image, edge_points, frame_count, center_radius=10,
                   center_color=(0, 255, 0)):  # Overlay green center
    center_y, center_x = np.mean(edge_points, axis=0).astype(int)  # Calculate the center of the edge points
    cv2.circle(output_image, (center_x, center_y), center_radius, center_color, -1)  # Draw a green dot at the center
    center_data = np.array([center_x, center_y, frame_count])  # Store center trace data as NumPy array
    return center_data, output_image


def detect_edges(image, frame_count, low_threshold=50, high_threshold=150):  # Detect edges and overlay center
    edges = cv2.Canny(image, low_threshold, high_threshold)  # Detect edges using Canny edge detection
    edge_points = np.column_stack(np.where(edges > 0))  # Find all edge points (non-zero pixels)
    output_image = overlay_edges(image, edge_points)  # Overlay edges on the image
    return overlay_center(output_image, edge_points, frame_count)  # Return center data and overlaid image


def detect_and_color_contours(image):  # Detect contours, color blobs, and return surface areas
    _, binary_image = cv2.threshold(image, 200, 255,
                                    cv2.THRESH_BINARY)  # Convert the grayscale image to binary (threshold)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale image to BGR to allow coloring
    surface_areas = []  # Initialize a list to store surface areas of blobs
    for contour in contours:  # Color the detected contours and calculate surface areas
        area = cv2.contourArea(contour)  # Get the area of the contour
        if area > 500:  # Only consider large blobs
            surface_areas.append(area)  # Append the area to the list
            cv2.drawContours(output_image, [contour], -1, (255, 105, 180),
                             thickness=cv2.FILLED)  # Draw the filled contour in pink
    return surface_areas, output_image  # Return the list of surface areas and the colored output image


def process_video(video_path, output_video_path, enhanced_video_output_path, frame_width, frame_height,
                  fps):  # Process the video
    video = read_video(video_path)  # Read input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec for output video
    final_output = cv2.VideoWriter(output_video_path, fourcc, fps,
                                   (frame_width, frame_height))  # Create VideoWriter object for output video
    enhanced_out = cv2.VideoWriter(enhanced_video_output_path, fourcc, fps,
                                   (frame_width, frame_height))  # Create VideoWriter for enhanced video
    center_data_list = []  # List to store center trace data
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_count = int(video.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number
        frame = cv2.resize(frame, (frame_width, frame_height))  # Ensure frame matches the expected dimensions
        gray_frame = convert_to_grayscale(frame)  # Convert to grayscale
        enhanced_frame = enhance_brightness(gray_frame)  # Enhance brightness
        surface_areas, counter_image = detect_and_color_contours(enhanced_frame)
        print(f"Frame {frame_count} - Surface areas of detected blobs: {surface_areas}")
        enhanced_out.write(counter_image)  # Write the enhanced frame to the enhanced video output
        center_data, edge_overlay_frame = detect_edges(enhanced_frame, frame_count)  # Detect edges and overlay center
        center_data_list.append(center_data)  # Append center data
        for i in range(1, len(center_data_list)):
            cv2.line(edge_overlay_frame, tuple(center_data_list[i - 1][:2]), tuple(center_data_list[i][:2]),
                     (255, 0, 0), 2)  # Draw full trace
        final_output.write(edge_overlay_frame)  # Write processed frame to output video

    video.release()  # Release video resources
    final_output.release()  # Release output video resources
    enhanced_out.release()  # Release enhanced video output resources
    print(f"Processed video saved as {output_video_path}")
    print(f"Enhanced video saved as {enhanced_video_output_path}")


def main():  # Main function to process video
    video_path = ask_user_for_video_path()  # Ask user for input video path
    if not video_path:
        print("No video selected. Exiting.")
        return

    output_video_path = os.path.join(os.path.dirname(video_path), '1_output_video.mp4')  # Get output video path
    enhanced_video_output_path = os.path.join(os.path.dirname(video_path),
                                              '2_enhanced_video_output.mp4')  # Get enhanced video output path
    video = read_video(video_path)  # Read video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()  # Release video after retrieving properties
    process_video(video_path, output_video_path, enhanced_video_output_path, frame_width, frame_height,
                  fps)  # Process the video frame by frame


if __name__ == "__main__":
    main()

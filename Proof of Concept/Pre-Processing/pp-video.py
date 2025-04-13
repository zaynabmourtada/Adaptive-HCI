import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor


def ask_user_for_video_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select the input video file",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    return file_path


def ask_user_for_output_paths(default_path):
    output_video = filedialog.asksaveasfilename(
        title="Save Processed Video",
        defaultextension=".mp4",
        initialfile="1_output_video.mp4",
        initialdir=os.path.dirname(default_path),
        filetypes=[("MP4 files", "*.mp4")]
    )
    enhanced_video = filedialog.asksaveasfilename(
        title="Save Enhanced Video",
        defaultextension=".mp4",
        initialfile="2_enhanced_video_output.mp4",
        initialdir=os.path.dirname(default_path),
        filetypes=[("MP4 files", "*.mp4")]
    )
    return output_video, enhanced_video


def read_video(video_path):  # Read the video from the given path
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        messagebox.showerror("Error", f"Error: Video not found or cannot be opened at {video_path}")
        exit()
    return video


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_brightness(image, cutoff=250, dark_factor=0.3, bright_factor=2.0):
    enhanced_image = image.copy()
    bright_pixels = image > cutoff
    enhanced_image[bright_pixels] = np.clip(image[bright_pixels] * bright_factor, 0, 255)
    dark_pixels = image <= cutoff
    enhanced_image[dark_pixels] = np.clip(image[dark_pixels] * dark_factor, 0, 255)
    return enhanced_image.astype(np.uint8)


def overlay_edges(image, edge_points, dot_radius=3, dot_color=(0, 0, 255)):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (y, x) in edge_points:
        cv2.circle(output_image, (x, y), dot_radius, dot_color, -1)
    return output_image


def overlay_center(output_image, edge_points, frame_count, center_radius=10, center_color=(0, 255, 0)):
    center_y, center_x = np.mean(edge_points, axis=0).astype(int)
    cv2.circle(output_image, (center_x, center_y), center_radius, center_color, -1)
    center_data = np.array([center_x, center_y, frame_count])
    return center_data, output_image


def detect_edges(image, frame_count, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    edge_points = np.column_stack(np.where(edges > 0))
    output_image = overlay_edges(image, edge_points)
    return overlay_center(output_image, edge_points, frame_count)


def detect_and_color_contours(image):
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    surface_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            surface_areas.append(area)
            cv2.drawContours(output_image, [contour], -1, (255, 105, 180), thickness=cv2.FILLED)
    return surface_areas, output_image


def process_frame(frame, frame_count, frame_width, frame_height, center_data_list):
    frame = cv2.resize(frame, (frame_width, frame_height))
    gray_frame = convert_to_grayscale(frame)
    enhanced_frame = enhance_brightness(gray_frame)
    surface_areas, contour_image = detect_and_color_contours(enhanced_frame)
    center_data, edge_overlay_frame = detect_edges(enhanced_frame, frame_count)
    
    # Append center data and draw trace
    center_data_list.append(center_data)
    for i in range(1, len(center_data_list)):
        cv2.line(edge_overlay_frame, tuple(center_data_list[i - 1][:2]), tuple(center_data_list[i][:2]), (255, 0, 0), 2)
    
    return surface_areas, contour_image, edge_overlay_frame


def process_video(video_path, output_video_path, enhanced_video_output_path, frame_width, frame_height, fps):
    video = read_video(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_output = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    enhanced_out = cv2.VideoWriter(enhanced_video_output_path, fourcc, fps, (frame_width, frame_height))
    
    center_data_list = []
    frame_count = 0
    
    with ThreadPoolExecutor() as executor:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            future = executor.submit(process_frame, frame, frame_count, frame_width, frame_height, center_data_list)
            surface_areas, contour_image, edge_overlay_frame = future.result()
            
            enhanced_out.write(contour_image)
            final_output.write(edge_overlay_frame)
    
    video.release()
    final_output.release()
    enhanced_out.release()
    print(f"Processed video saved as {output_video_path}")
    print(f"Enhanced video saved as {enhanced_video_output_path}")


def main():
    video_path = ask_user_for_video_path()
    if not video_path:
        print("No video selected. Exiting.")
        return
    
    output_video_path, enhanced_video_output_path = ask_user_for_output_paths(video_path)
    
    video = read_video(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    
    process_video(video_path, output_video_path, enhanced_video_output_path, frame_width, frame_height, fps)


if __name__ == "__main__":
    main()

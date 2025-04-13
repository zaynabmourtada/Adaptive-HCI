<<<<<<< HEAD
# Soham Naik - Senior Design - Adaptive HCI - Pre-Processing & Image Edge Detection DEMO - 10/17/2024

import cv2
import numpy as np
import os

def get_image_path(filename):  # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def read_image(image_path):  # Read the image from the given path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        exit()
    return image

def convert_to_grayscale(image):  # Convert the input image to grayscale
    output_image = image.copy()
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

def enhance_brightness(image, cutoff=250, dark_factor=0.3, bright_factor=2.0):  # Enhance brightness of the image
    output_image = image.copy()  # Create a copy of the image to enhance brightness
    bright_pixels = image > cutoff  # Amplify bright areas (pixels above the cutoff)
    output_image[bright_pixels] = np.clip(image[bright_pixels] * bright_factor, 0, 255)
    dark_pixels = image <= cutoff  # Darken dark areas (pixels below or equal to the cutoff)
    output_image[dark_pixels] = np.clip(image[dark_pixels] * dark_factor, 0, 255)
    return output_image.astype(np.uint8)

def overlay_edges(image, edge_points, dot_radius=3, dot_color=(0, 0, 255)):  # Overlay red edges on the image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale image to BGR
    for (y, x) in edge_points:
        cv2.circle(output_image, (x, y), dot_radius, dot_color, -1)  # Draw red dots at edge points
    return output_image

def overlay_center(image, edge_points, center_radius=10, center_color=(0, 255, 0)):  # Overlay green center
    output_image = image.copy()  # Create a copy of the image to enhance brightness
    center_y, center_x = np.mean(edge_points, axis=0).astype(int)  # Calculate the center of the edge points
    cv2.circle(output_image, (center_x, center_y), center_radius, center_color, -1)  # Draw a green dot at the center
    return output_image

def detect_edges(image, low_threshold=50, high_threshold=150):  # Detect edges and overlay center
    edges = cv2.Canny(image, low_threshold, high_threshold)  # Detect edges using Canny edge detection
    edge_points = np.column_stack(np.where(edges > 0))  # Find all edge points (non-zero pixels)
    return edge_points

def save_image(image, filename):  # Save the processed image
    output_path = get_image_path(filename)
    cv2.imwrite(output_path, image)  # Save the image to the script's directory
    print(f"Image saved at {output_path}")

def main():  # Main function to process image
    image_path = get_image_path('image/0_input_image.jpg')  # Get input image path
    image = read_image(image_path)  # Read the input image

    gray_image = convert_to_grayscale(image)  # Convert image to grayscale
    enhanced_image = enhance_brightness(gray_image)  # Enhance brightness of the grayscale image
    edge_points = detect_edges(enhanced_image)  # Apply edge detection and overlay edges
    overlay_edges_img = overlay_edges(enhanced_image, edge_points) # Overlay edges on the image
    overlay_center_img = overlay_center(overlay_edges_img, edge_points)  # Overlay center on the image

    # Save intermediate and final output images
    save_image(gray_image, 'image/1_gray_image.jpg')
    save_image(enhanced_image, 'image/2_enhanced_image.jpg')
    save_image(overlay_edges_img, 'image/3_edge_overlay_image.jpg')
    save_image(overlay_center_img, 'image/4_center_overlay_image.jpg')

if __name__ == "__main__":
    main()
=======
# Soham Naik - Senior Design - Adaptive HCI - Pre-Processing & Image Edge Detection DEMO - 10/17/2024

import cv2
import numpy as np
import os

def get_image_path(filename):  # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def read_image(image_path):  # Read the image from the given path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        exit()
    return image

def convert_to_grayscale(image):  # Convert the input image to grayscale
    output_image = image.copy()
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

def enhance_brightness(image, cutoff=250, dark_factor=0.3, bright_factor=2.0):  # Enhance brightness of the image
    output_image = image.copy()  # Create a copy of the image to enhance brightness
    bright_pixels = image > cutoff  # Amplify bright areas (pixels above the cutoff)
    output_image[bright_pixels] = np.clip(image[bright_pixels] * bright_factor, 0, 255)
    dark_pixels = image <= cutoff  # Darken dark areas (pixels below or equal to the cutoff)
    output_image[dark_pixels] = np.clip(image[dark_pixels] * dark_factor, 0, 255)
    return output_image.astype(np.uint8)

def overlay_edges(image, edge_points, dot_radius=3, dot_color=(0, 0, 255)):  # Overlay red edges on the image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale image to BGR
    for (y, x) in edge_points:
        cv2.circle(output_image, (x, y), dot_radius, dot_color, -1)  # Draw red dots at edge points
    return output_image

def overlay_center(image, edge_points, center_radius=10, center_color=(0, 255, 0)):  # Overlay green center
    output_image = image.copy()  # Create a copy of the image to enhance brightness
    center_y, center_x = np.mean(edge_points, axis=0).astype(int)  # Calculate the center of the edge points
    cv2.circle(output_image, (center_x, center_y), center_radius, center_color, -1)  # Draw a green dot at the center
    return output_image

def detect_edges(image, low_threshold=50, high_threshold=150):  # Detect edges and overlay center
    edges = cv2.Canny(image, low_threshold, high_threshold)  # Detect edges using Canny edge detection
    edge_points = np.column_stack(np.where(edges > 0))  # Find all edge points (non-zero pixels)
    return edge_points

def save_image(image, filename):  # Save the processed image
    output_path = get_image_path(filename)
    cv2.imwrite(output_path, image)  # Save the image to the script's directory
    print(f"Image saved at {output_path}")

def main():  # Main function to process image
    image_path = get_image_path('image/0_input_image.jpg')  # Get input image path
    image = read_image(image_path)  # Read the input image

    gray_image = convert_to_grayscale(image)  # Convert image to grayscale
    enhanced_image = enhance_brightness(gray_image)  # Enhance brightness of the grayscale image
    edge_points = detect_edges(enhanced_image)  # Apply edge detection and overlay edges
    overlay_edges_img = overlay_edges(enhanced_image, edge_points) # Overlay edges on the image
    overlay_center_img = overlay_center(overlay_edges_img, edge_points)  # Overlay center on the image

    # Save intermediate and final output images
    save_image(gray_image, 'image/1_gray_image.jpg')
    save_image(enhanced_image, 'image/2_enhanced_image.jpg')
    save_image(overlay_edges_img, 'image/3_edge_overlay_image.jpg')
    save_image(overlay_center_img, 'image/4_center_overlay_image.jpg')

if __name__ == "__main__":
    main()
>>>>>>> origin
# END
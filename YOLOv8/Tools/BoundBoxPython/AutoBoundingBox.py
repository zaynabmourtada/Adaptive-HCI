import os
import cv2
import numpy as np

def process_images(input_folder, output_folder, labels_folder=None, output_class_id=0, threshold_value=100):
    # Ensure the output directories exist
    os.makedirs(output_folder, exist_ok=True)
    
    # If labels_folder is not provided, use output_folder for labels
    if labels_folder is None:
        labels_folder = output_folder
    else:
        os.makedirs(labels_folder, exist_ok=True)

    # Gather all image files in the input folder
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.splitext(f)[1].lower() in valid_exts]

    for image_name in image_files:
        image_path = os.path.join(input_folder, image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read {image_path}. Skipping...")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold: Anything darker than `threshold_value` becomes black (0)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Find coordinates of all non-zero (white) pixels
        non_zero_pixels = cv2.findNonZero(thresh)
        
        if non_zero_pixels is not None and len(non_zero_pixels) > 0:
            # Get bounding box directly from non-zero pixels
            x, y, w, h = cv2.boundingRect(non_zero_pixels)

            # Draw the bounding box on the original image
            image_with_box = image.copy()
            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Convert thresholded image to BGR for drawing colored bounding box
            thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # Draw the bounding box on the thresholded image
            thresh_with_box = thresh_bgr.copy()
            cv2.rectangle(thresh_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get image dimensions
            img_h, img_w = image.shape[:2]

            # Compute normalized center, width, height for YOLO format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Build the output text file path
            txt_filename = os.path.splitext(image_name)[0] + '.txt'
            txt_path = os.path.join(labels_folder, txt_filename)

            # Write bounding box to .txt (YOLO format)
            with open(txt_path, 'w') as f:
                f.write(f"{output_class_id} {x_center:.6f} {y_center:.6f} "
                        f"{w_norm:.6f} {h_norm:.6f}\n")
            
            #print(f"Found {len(non_zero_pixels)} non-zero pixels in {image_name}")
            #print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
            
            # Save the original image with bounding box to output folder
            #output_image_path = os.path.join(output_folder, image_name)
            #cv2.imwrite(output_image_path, image_with_box)

            # Save the thresholded image with bounding box
            thresh_filename = "thresh_" + image_name
            thresh_output_path = os.path.join(output_folder, thresh_filename)
            cv2.imwrite(thresh_output_path, thresh_with_box)
            
            # Print results
            # print(f"YOLO label saved at: {txt_path}")   
        else:
            print(f"No non-black pixels detected in {image_name}")

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the input and output folders
    input_folder = os.path.join(script_dir, "Input/User_1")
    output_folder = os.path.join(script_dir, "OUTPUT/IMAGES/User_1")
    labels_folder = os.path.join(script_dir, "OUTPUT/LABELS/User_1")  # Optional: separate folder for labels
    
    # Process the images
    process_images(input_folder, output_folder, labels_folder, output_class_id=0, threshold_value=8)
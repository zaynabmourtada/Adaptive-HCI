import os
import tensorflow as tf

# Define the input directory where TensorFlow models are stored
input_dir = r"F:\GitHub\Adaptive-HCI\YOLOv8\Models\yolo_v2\weights"

# Loop through all folders in the directory
for folder_name in os.listdir(input_dir):
    tf_model_path = os.path.join(input_dir, folder_name)

    # Check if the folder contains a valid TensorFlow SavedModel
    if os.path.isdir(tf_model_path) and os.path.exists(os.path.join(tf_model_path, "saved_model.pb")):
        tflite_path = os.path.join(input_dir, folder_name + ".tflite")  # Output TFLite filename

        # Skip conversion if the .tflite file already exists
        if os.path.exists(tflite_path):
            print(f"Skipping {folder_name} (TFLite model already exists)")
            continue

        print(f"Converting {folder_name} to {tflite_path}...")

        # Load the TensorFlow model
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

        # Convert to TensorFlow Lite
        tflite_model = converter.convert()

        # Save the converted model
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"Successfully converted: {tflite_path}")

print("All TensorFlow models converted to TFLite.")

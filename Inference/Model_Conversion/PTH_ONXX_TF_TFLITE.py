import torch
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def convert_pth_to_tflite(pth_path, onnx_path, saved_model_dir, tflite_path):
    # ========================
    # 1. Export PyTorch Model to ONNX
    # ========================
    # Load your PyTorch model
    model = torch.load(pth_path, map_location=torch.device('cpu'))
    model.eval()

    # Create a dummy input based on your model's expected input shape.
    # Adjust the dimensions as required (e.g., [batch_size, channels, height, width])
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export the model to ONNX format
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=['input'], 
        output_names=['output'],
        opset_version=11
    )
    print(f"ONNX model saved to: {onnx_path}")

    # ========================
    # 2. Convert ONNX Model to TensorFlow SavedModel
    # ========================
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert ONNX model to TensorFlow representation
    tf_rep = prepare(onnx_model)
    
    # Export the TensorFlow model as a SavedModel
    tf_rep.export_graph(saved_model_dir)
    print(f"TensorFlow SavedModel exported to: {saved_model_dir}")

    # ========================
    # 3. Convert TensorFlow SavedModel to TensorFlow Lite Model
    # ========================
    # Create the TFLiteConverter from the SavedModel directory
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Optional: Enable optimizations (such as quantization) if needed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model to TensorFlow Lite format
    tflite_model = converter.convert()
    
    # Save the TFLite model to a file
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TensorFlow Lite model saved to: {tflite_path}")

if __name__ == "__main__":
    # Define your file paths
    pth_path = "digit_recognizer_finetuned.pth"
    onnx_path = "digit_recognizer_finetuned.onnx"
    saved_model_dir = "saved_model"
    tflite_path = "digit_recognizer_finetuned.tflite"
    
    convert_pth_to_tflite(pth_path, onnx_path, saved_model_dir, tflite_path)

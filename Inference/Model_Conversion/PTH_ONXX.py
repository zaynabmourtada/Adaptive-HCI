import torch
import torch.onnx

# File paths
pth_path = "digit_recognizer_finetuned.pth"
onnx_path = "digit_recognizer_finetuned.onnx"

# ========================
# 1. Load the Entire Model
# ========================
# Load the model architecture and weights
model = torch.load(pth_path, map_location='cpu')
model.eval()

# ========================
# 2. Create Dummy Input
# ========================
# Automatically detect input shape by inspecting the first layer
# Try common input sizes if unsure
try:
    # If the model has a defined input shape, use it
    dummy_input = next(model.parameters()).unsqueeze(0)
except StopIteration:
    # Fallback to a common input shape for image models
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust as needed

# ========================
# 3. Export to ONNX
# ========================
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

print(f"ONNX model saved to: {onnx_path}")

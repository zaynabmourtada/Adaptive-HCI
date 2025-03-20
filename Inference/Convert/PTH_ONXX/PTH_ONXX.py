import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F

# ========================
# Model Definition
# ========================
class LetterRecognizer(nn.Module):
    def __init__(self):
        super(LetterRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(3*3*128, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 3*3*128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========================
# File Paths
# ========================
pth_path = "resnet18_custom.pth"
onnx_path = "resnet18_custom.onnx"

# ========================
# 1. Load the Entire Model
# ========================
model = torch.load(pth_path, map_location='cpu')
model.eval()

# ========================
# 2. Create Dummy Input
# ========================
dummy_input = torch.randn(1, 1, 28, 28)  # Grayscale (1 channel) 28x28 images

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

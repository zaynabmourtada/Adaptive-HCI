# Author: Zaynab Mourtada 
# Purpose: Load a trained letter recognition model and evaluate its accuracy on two test datasets
# Last Modified: 4/15/2025
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.serialization

class UppercaseLetterRecognizer(nn.Module):
    def __init__(self):
        super(UppercaseLetterRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 26)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformations 
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load trained model
def load_model(model_path, device):
    torch.serialization.add_safe_globals([UppercaseLetterRecognizer])
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model

# Evaluate model performance
def evaluate_model(model, test_loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set paths to data and model files based on where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    chars74k_path= os.path.join(project_root, "Training Data", "Chars74K")
    xamera_path = os.path.join(project_root, "Training Data", "Xamera Letter Dataset") 
    model_path = os.path.join(project_root, "Trained Models", "PTH Files", "letter_recognizer_finetuned.pth")
    test_idx_path = os.path.join(base_dir, "test_idx.pth")

    if not os.path.exists(test_idx_path):
        raise FileNotFoundError(f"Test indices file not found at {test_idx_path}")
    
    chars74k_test_idx, xamera_test_idx = torch.load(test_idx_path)
    print(f"Loaded test indices from {test_idx_path}")
    
    chars74k_dataset = ImageFolder(root=chars74k_path, transform=test_transform)
    xamera_dataset=ImageFolder(root=xamera_path, transform=test_transform)
    
    # Create test subsets using pre-saved indices
    chars74k_test = torch.utils.data.Subset(chars74k_dataset, chars74k_test_idx)
    xamera_test = torch.utils.data.Subset(xamera_dataset, xamera_test_idx)

    chars74k_loader = DataLoader(chars74k_test, batch_size=128, shuffle=False)
    xamera_loader = DataLoader(xamera_test, batch_size=128, shuffle=False)

    model = load_model(model_path, device)
    print(f"Loaded model from {model_path}")

    print("Evaluating Chars74K Dataset:")
    evaluate_model(model, chars74k_loader, device)

    print("Evaluating Xamera Dataset:")
    evaluate_model(model, xamera_loader, device)

if __name__ == "__main__":
    main()

# Author: Zaynab Mourtada 
# Purpose: Load a trained digit recognition model and evaluate its accuracy on three test datasets
# Last Modified: 4/20/2025
import os
import sys  
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, MNIST
from torch.utils.data import DataLoader

# To ensure imports from the training script work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from train_digit_recognizer import ImprovedDigitRecognizer, mnist_transform, common_transform

# Xamera transformation
xamera_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
def load_model(model_path, device):
    torch.serialization.add_safe_globals([ImprovedDigitRecognizer])
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model

# Evaluate model performance
def test_model(model, test_loader, dataset_name, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy on {dataset_name}: {accuracy:.2f}%\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set paths to data and model files based on where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    dida_path = os.path.join(project_root, "Training Data", "10000 DIDA")
    xamera_path = os.path.join(project_root, "Training Data", "Xamera Digit Dataset")
    model_path = os.path.join(project_root, "Trained Models", "PTH Files", "digit_recognizer_finetuned.pth")
    
    # Load MNIST test dataset
    mnist_test_dataset = MNIST(root="./data", train=False, transform=mnist_transform, download=True)
    
    # Load DIDA test dataset
    if not os.path.exists(dida_path):
        print(f"Error: Dataset path not found: {dida_path}")
        return
    dida_test_dataset = ImageFolder(root=dida_path, transform=common_transform)
    
    # Load Xamera test dataset with transformations
    if not os.path.exists(xamera_path):
        raise FileNotFoundError(f"Xamera dataset not found at {xamera_path}")
    xamera_test_dataset = ImageFolder(root=xamera_path, transform=xamera_transform)
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path, device)
    print(f"Loaded model from {model_path}")
    
    # Evaluate on all test datasets
    datasets = {
        "MNIST": mnist_test_dataset,
        "DIDA": dida_test_dataset,
        "Xamera": xamera_test_dataset,
    }

    for name, dataset in datasets.items():
        test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        print(f"Testing {name} Dataset:")
        test_model(model, test_loader, name, device)

if __name__ == "__main__":
    main()

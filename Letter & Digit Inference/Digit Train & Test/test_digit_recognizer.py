import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, MNIST
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from train_digit_recognizer import ImprovedDigitRecognizer, mnist_transform, common_transform

def test_model(model, test_loader, dataset_name, device):
    model.eval()
    correct = 0
    total = 0
    sample_images = []
    sample_labels = []
    sample_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if len(sample_images) < 10:  
                sample_images.extend(images.cpu().numpy())
                sample_labels.extend(labels.cpu().numpy())
                sample_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy on {dataset_name}: {accuracy:.2f}%\n")

    show_sample_images(sample_images[:10], sample_labels[:10], sample_predictions[:10])

def show_sample_images(images, labels, predictions):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        ax.axis("off")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_test_dataset = MNIST(root="./data", train=False, transform=mnist_transform, download=True)

    dida_path = r"C:\Users\zayna\OneDrive\Documents\University\Senior Design\adaptive_code\Adaptive-HCI\Inference\10000 DIDA"
    if not os.path.exists(dida_path):
        print(f"Error: Dataset path not found: {dida_path}")
        return
    
    dida_test_dataset = ImageFolder(root=dida_path, transform=common_transform)

    #xamera_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    xamera_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    xamera_path = r"C:\Users\zayna\OneDrive\Documents\University\Senior Design\adaptive_code\Adaptive-HCI\Inference\Xamera Dataset"
    xamera_test_dataset = ImageFolder(root=xamera_path, transform=xamera_transform)

    datasets = {
        "MNIST": mnist_test_dataset,
        "DIDA": dida_test_dataset,
        "Xamera": xamera_test_dataset,
    }

    model_path = os.path.join(script_dir, "digit_recognizer_finetuned.pth")
    model = ImprovedDigitRecognizer().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("Error: Model file not found! Ensure digit_recognizer.pth exists.")
        return

    for name, dataset in datasets.items():
        test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        test_model(model, test_loader, name, device)

if __name__ == "__main__":
    main()

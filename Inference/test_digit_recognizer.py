import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class ImprovedDigitRecognizer(torch.nn.Module):
    def __init__(self):
        super(ImprovedDigitRecognizer, self).__init__()
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

def test_model(model, test_loader, device):
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
    print(f"Test Accuracy: {accuracy:.2f}%\n")

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dataset_path = os.path.join(script_dir, "test_dataset.pth")

    if os.path.exists(test_dataset_path):
        test_indices = torch.load(test_dataset_path)  
        print("Test dataset indices loaded successfully!")
    else:
        print("Error: Test dataset file not found!")
        return
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_path = os.path.join(script_dir, "10000 DIDA")  
    full_dataset = ImageFolder(root=dataset_path, transform=transform)

    if max(test_indices) >= len(full_dataset):
        print("Error: Test indices exceed dataset size!")
        return
    
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_path = os.path.join(script_dir, "digit_recognizer.pth")
    model = ImprovedDigitRecognizer().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("Error: Model file not found!")
        return

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()

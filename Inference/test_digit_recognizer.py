import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import os

class DigitRecognizer(torch.nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(7*7*32, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 7*7*32)
        x = torch.relu(self.fc1(x))
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
    device = torch.device("cpu")  

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = MNIST(root='./MNIST_data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = DigitRecognizer().to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "digit_recognizer.pth")

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()

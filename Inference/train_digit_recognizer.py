import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


class ImprovedDigitRecognizer(nn.Module):
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

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(30): 
        total_loss = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / len(train_loader.dataset):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=5, shear=5, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_path = "/home/zaynabmo/inference_project/digit_model/10000 DIDA"
    full_dataset = ImageFolder(root=dataset_path, transform=transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    def validate_model(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item()
                correct += (output.argmax(1)==labels).sum().item()
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {100 * correct / len(val_loader.dataset):.4f}")

    test_dataset_path = os.path.join(os.getcwd(), 'test_dataset.pth')
    test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else list(range(len(test_dataset)))
    torch.save(test_indices, test_dataset_path)


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

    model = ImprovedDigitRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, optimizer, criterion, device)
    validate_model(model, val_loader, criterion, device)

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_recognizer.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    main()

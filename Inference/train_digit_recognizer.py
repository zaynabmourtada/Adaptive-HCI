import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import ImageFolder, MNIST
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def invert_mnist(img):
        return TF.invert(img)
    
mnist_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.Lambda(invert_mnist), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

common_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=5, shear=5, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

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

def train_model(model, train_loader, optimizer, criterion, scheduler, device):
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

        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / len(train_loader.dataset):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DIDA dataset
    dataset_path = "/home/zaynabmo/inference_project/digit_model/10000 DIDA"
    dida_dataset = ImageFolder(root=dataset_path, transform=common_transform)

    # Load MNIST dataset
    mnist_dataset = MNIST(root="./data", train=True, transform=mnist_transform, download=True)

    #xamera_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    xamera_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(5),  # Increase rotation range
    transforms.RandomAffine(degrees=5, shear=2, translate=(0.05, 0.05)),  # Add random translations
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Simulate lighting changes
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Introduce slight distortions  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

    xamera_path = "/home/zaynabmo/inference_project/digit_model/Xamera Dataset"
    xamera_dataset = ImageFolder(root=xamera_path, transform=xamera_transform)

    # Combine both datasets
    full_dataset = ConcatDataset([mnist_dataset, dida_dataset, xamera_dataset])

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Added these
    train_indices = set(train_dataset.indices) if hasattr(train_dataset, 'indices') else set(range(len(train_dataset)))
    test_indices = set(test_dataset.indices) if hasattr(test_dataset, 'indices') else set(range(len(test_dataset)))
    assert len(train_indices & test_indices) == 0, "Overlap detected between train and test sets! Data leakage risk!"
    print("No overlap between training and testing data.")
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

    #test_dataset_path = os.path.join(os.getcwd(), 'test_dataset.pth')
    #test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else list(range(len(test_dataset)))
    #torch.save(test_indices, test_dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

    model = ImprovedDigitRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images, Testing on {len(test_dataset)} images")

    train_model(model, train_loader, optimizer, criterion, scheduler, device)
    validate_model(model, val_loader, criterion, device)

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_recognizer.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    # Fine-Tune on Only Xamera Dataset
    fine_tune_epochs = 10
    fine_tune_loader = DataLoader(xamera_dataset, batch_size=128, shuffle=True, drop_last=False)

    fine_tune_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print("\nFine-Tuning on Xamera Dataset...")
    for epoch in range(fine_tune_epochs):
        model.train()
        total_loss = 0
        correct = 0
        for images, labels in fine_tune_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()

        fine_tune_scheduler.step()
        print(f"Fine-Tune Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / len(xamera_dataset):.4f}%")

    # Save the fine-tuned model
    fine_tune_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_recognizer_finetuned.pth")
    torch.save(model.state_dict(), fine_tune_model_path)
    print(f"Fine-Tuned Model saved at {fine_tune_model_path}")

if __name__ == "__main__":
    main()

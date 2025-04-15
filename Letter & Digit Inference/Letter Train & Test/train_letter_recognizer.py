# Author: Zaynab Mourtada 
# Purpose: Train and fine-tune a CNN to recognize uppercase letters from image datasets
# Last Modified: 4/15/2025
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import ImageFolder
import torch.optim as optim
import os

# Transformations
common_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=5, shear=5, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

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
        self.dropout = nn.Dropout(0.5)
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

# Trains, validates, and saves best model 
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, save_path):
    best_val_acc = 0.0
    for epoch in range(25):
        model.train()
        total_loss, correct = 0, 0
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
        train_acc = 100 * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item()
                val_correct += (output.argmax(1) == labels).sum().item()
        val_acc = 100 * val_correct / len(val_loader.dataset)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with Val Accuracy: {val_acc:.2f}%")
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define paths based on where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    chars74k_path = os.path.join(project_root, "Training Data", "Chars74K")
    xamera_path   = os.path.join(project_root, "Training Data", "Xamera Letter Dataset")
    model_save_path = os.path.join(base_dir, "letter_recognizer.pth")
    fine_tune_model_path = os.path.join(base_dir, "letter_recognizer_finetuned.pth")

    chars74k_dataset = ImageFolder(root=chars74k_path, transform=common_transform)
    xamera_dataset = ImageFolder(root=xamera_path, transform=common_transform)
    full_dataset = ConcatDataset([chars74k_dataset, xamera_dataset])
    
    # Combine datasets and split into train/val/test
    train_size = int(0.8* len(full_dataset))
    val_size = int(0.1*len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset= random_split(full_dataset,[train_size, val_size, test_size])

    # Save test indices
    chars74k_test_idx = [i for i in test_dataset.indices if i < len(chars74k_dataset)]
    xamera_test_idx = [i-len(chars74k_dataset) for i in test_dataset.indices if i>=len(chars74k_dataset)]
    torch.save((chars74k_test_idx, xamera_test_idx), "test_idx.pth")
    print(f"Test indices are saved.")

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

    model = UppercaseLetterRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=20)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images")

    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, model_save_path)

    # Fine-Tuning Model 
    print("\nFine-Tuning on Xamera Dataset...")
    
    # Freeze conv and batch norm layers
    for layer in [model.conv1, model.bn1, model.conv2, model.bn2, model.conv3, model.bn3]:
        for param in layer.parameters():
            param.requires_grad = False
    print("Froze convolutional and batch norm layers for fine-tuning.")

    fine_tune_loader=DataLoader(xamera_dataset, batch_size=128, shuffle=True, drop_last=False)
    fine_tune_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(10):
        total_loss, correct=0, 0
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

    torch.save(model, fine_tune_model_path)
    print(f"Fine-Tuned Model saved at {fine_tune_model_path}")

if __name__=="__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch.optim as optim
import numpy as np
import os
import pandas as pd


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
    
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, save_path):
    best_val_acc = 0.0
    for epoch in range(25):
        model.train()
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
        train_acc = 100 * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item()
                val_correct += (output.argmax(1) == labels).sum().item()
        val_acc = 100 * val_correct / len(val_loader.dataset)
        print(f"          Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with Val Accuracy: {val_acc:.2f}% → {save_path}")
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chars74k_path = r"C:\Users\zayna\OneDrive\Documents\University\Senior Design\GitHub code\Adaptive-HCI\Letter & Digit Inference\Training Data\Chars74K"
    xamera_path = r"C:\Users\zayna\OneDrive\Documents\University\Senior Design\GitHub code\Adaptive-HCI\Letter & Digit Inference\Training Data\Xamera Letter Dataset"
    chars74k_dataset = ImageFolder(root=chars74k_path, transform=common_transform)
    xamera_dataset = ImageFolder(root=xamera_path, transform=common_transform)

    full_dataset = ConcatDataset([chars74k_dataset, xamera_dataset])
    train_size = int(0.8* len(full_dataset))
    val_size = int(0.1*len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset= random_split(full_dataset,[train_size, val_size, test_size])

    chars74k_test_idx = [i for i in test_dataset.indices if i < len(chars74k_dataset)]
    xamera_test_idx = [i-len(chars74k_dataset) for i in test_dataset.indices if i>=len(chars74k_dataset)]
    torch.save((chars74k_test_idx, xamera_test_idx), "test_idx.pth")
    print(f"Test indices saved!")

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)

    model = UppercaseLetterRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=20)

    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images")

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letter_recognizer.pth")
    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, model_save_path)

    fine_tune_loader=DataLoader(xamera_dataset, batch_size=128, shuffle=True, drop_last=False)
    fine_tune_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("\nFine-Tuning on Xamera Dataset...")

    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.conv3.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.bn2.parameters():
        param.requires_grad = False
    for param in model.bn3.parameters():
        param.requires_grad = False

    print("Frozen convolutional and batch norm layers for fine-tuning.")

    for epoch in range(10):
        total_loss=0
        correct=0
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

    fine_tune_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "letter_recognizer_finetuned.pth")
    torch.save(model, fine_tune_model_path)
    print(f"Fine-Tuned Model saved at {fine_tune_model_path}")

if __name__=="__main__":
    main()

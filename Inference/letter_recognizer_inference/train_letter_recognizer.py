import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms 
<<<<<<< HEAD
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
=======
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
>>>>>>> cc7f70c9b64a4701a0165e4ec0965ac3d9948527
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import KFold
import pandas as pd
<<<<<<< HEAD
import tkinter as tk
from tkinter import filedialog

def invert_emnist(img):
    return TF.invert(img)  
=======
import random
>>>>>>> cc7f70c9b64a4701a0165e4ec0965ac3d9948527

emnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CSV Dataset that loads letters from a CSV file with label adjustment.
class CSVLetterDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, header=None).values
<<<<<<< HEAD
        
        # Get the raw labels and check their range.
        raw_labels = self.data[:, 0].astype(np.int32)
        print(f"Loading {csv_path} - raw labels range: {raw_labels.min()} to {raw_labels.max()}")
        
        # If labels are 1-indexed (i.e. min == 1), subtract 1 to make them 0-indexed.
        if raw_labels.min() == 1:
            self.labels = raw_labels - 1
        else:
            self.labels = raw_labels
            
        self.images = self.data[:, 1:].reshape(-1, 28, 28)
=======
        self.labels = self.data[:, 0].astype(int)

        if self.labels.min() == 1:
            print(f"Fixing EMNIST label range in {csv_path}")
            self.labels -= 1 

        self.images = self.data[:, 1:].reshape(-1, 28, 28).astype('float32')  
>>>>>>> cc7f70c9b64a4701a0165e4ec0965ac3d9948527
        self.images = self.images.astype('float32') / 255.0
        self.images = torch.tensor(self.images).unsqueeze(1)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        if self.labels.min() < 0 or self.labels.max() >= 26:
            print(f"ERROR: Label out of range in {csv_path}! Min={self.labels.min()}, Max={self.labels.max()}")
            print(f"Unique Labels: {np.unique(self.labels.numpy())}")
            exit(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class LetterRecognizer(nn.Module):
    def __init__(self):
        super(LetterRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(3*3*128, 256)
        self.fc2 = nn.Linear(256, 26)
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

def train_model(model, train_loader, optimizer, criterion, scheduler, device, epochs=30):
    model.train()
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")

<<<<<<< HEAD
def k_fold_train(model, dataset, k=5, epochs=10, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nStarting Fold {fold+1}/{k}...")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = LetterRecognizer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        train_model(model, train_loader, optimizer, criterion, scheduler, device, epochs=epochs)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        fold_accuracies.append(val_acc)
        print(f"Fold {fold+1} Validation Accuracy: {val_acc:.2f}%")

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nFinal K-Fold Accuracy: {avg_accuracy:.2f}%")
    return model

def select_csv_file(prompt):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV files", "*.csv")])
    root.destroy()
    return file_path
=======
>>>>>>> cc7f70c9b64a4701a0165e4ec0965ac3d9948527

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Let the user select the CSV files.
    print("Select the EMNIST letters train CSV file")
    emnist_train_csv = select_csv_file("Select the EMNIST letters train CSV file")
    print("Select the EMNIST letters test CSV file")
    emnist_test_csv = select_csv_file("Select the EMNIST letters test CSV file")
    print("Select the Xamera letters train CSV file")
    xamera_train_csv = select_csv_file("Select the Xamera letters train CSV file")
    print("Select the Xamera letters test CSV file")
    xamera_test_csv = select_csv_file("Select the Xamera letters test CSV file")

    emnist_train_dataset = CSVLetterDataset(emnist_train_csv)
    emnist_test_dataset = CSVLetterDataset(emnist_test_csv)
    xamera_train_dataset = CSVLetterDataset(xamera_train_csv)
    xamera_test_dataset = CSVLetterDataset(xamera_test_csv)

<<<<<<< HEAD
    # Create a train-validation split from the EMNIST training dataset.
=======


    # Train-Validation split from EMNIST training dataset
>>>>>>> cc7f70c9b64a4701a0165e4ec0965ac3d9948527
    train_size = int(0.9 * len(emnist_train_dataset))
    val_size = len(emnist_train_dataset) - train_size
    emnist_train_subset, emnist_val_subset = random_split(emnist_train_dataset, [train_size, val_size])

    train_loader = DataLoader(emnist_train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(emnist_val_subset, batch_size=128, shuffle=False)

    print("Training on EMNIST Data First...")
    model = LetterRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, optimizer, criterion, scheduler, device)

<<<<<<< HEAD
    print("\nFine-tuning on Xamera Data...")
    model = k_fold_train(model, xamera_train_dataset, k=5, epochs=10, batch_size=128)
=======
    print("\nFine-Tuning on Xamera Dataset...")
    fine_tune_epochs = 20
    fine_tune_loader = DataLoader(xamera_train_dataset, batch_size=128, shuffle=True, drop_last=False)
>>>>>>> cc7f70c9b64a4701a0165e4ec0965ac3d9948527

    fine_tune_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
        print(f"Fine-Tune Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / len(xamera_train_dataset):.4f}%")
    
    fine_tune_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letter_recognizer_finetuned.pth")
    torch.save(model, fine_tune_model_path)
    print(f"Fine-Tuned Model saved at {fine_tune_model_path}")

if __name__ == "__main__":
    main()
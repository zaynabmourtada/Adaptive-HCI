import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_letter_recognizer import LetterRecognizer, CSVLetterDataset

def show_sample_images(images, labels, predictions):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        ax.axis("off")
    plt.show()

def test_model(model, test_loader, criterion, device, dataset_name):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    sample_images = []
    sample_labels = []
    sample_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if len(sample_images) < 10:  
                sample_images.extend(images.cpu().numpy())
                sample_labels.extend(labels.cpu().numpy())
                sample_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total_samples
    avg_loss = total_loss / len(test_loader)

    print(f"Test Accuracy on {dataset_name}: {accuracy:.2f}%\n")

    show_sample_images(sample_images[:10], sample_labels[:10], sample_predictions[:10])
    return avg_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letter_recognizer_finetuned.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = LetterRecognizer().to(device)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")


    emnist_test_dataset = CSVLetterDataset('EMNIST Letters/emnist-letters-test.csv') # Unzip the EMNIST Letters folder
    xamera_test_dataset = CSVLetterDataset('xamera-letters-test.csv')

    emnist_test_loader = DataLoader(emnist_test_dataset, batch_size=128, shuffle=False)
    xamera_test_loader = DataLoader(xamera_test_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    print("\nTesting on EMNIST Dataset...")
    emnist_loss, emnist_acc = test_model(model, emnist_test_loader, criterion, device, "EMNIST")

    print("\nTesting on Xamera Dataset...")
    xamera_loss, xamera_acc = test_model(model, xamera_test_loader, criterion, device, "Xamera")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# -------------------------------
# 1. Define a smaller CNN model
# -------------------------------
class SmallCNN(nn.Module):
    """
    A simple CNN for 28x28 grayscale images.
    Adjust the number of convolution layers, feature maps, or
    fully connected layers as needed.
    """
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        # in_channels=1 because we have grayscale images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After two convolutions + two 2×2 pools:
        # Each pool halves the spatial dimension:
        # 28 -> 14 -> 7
        # So final feature map is 64×7×7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)             # Flatten to [batch, 3136]
        x = F.relu(self.fc1(x))               # [batch, 128]
        x = self.fc2(x)                       # [batch, num_classes]
        return x

# -------------------------------
# 2. (Optional) Model Exporter
# -------------------------------
class ModelExporter:
    def __init__(self, model, class_names):
        """
        Args:
            model: Trained PyTorch model.
            class_names: List of class names.
        """
        self.model = model
        self.class_names = class_names

    def save(self, file_path):
        """Save the model's state dictionary along with class names."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names
        }
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")

def main():
    # -------------------------------
    # 3. Define transforms
    # -------------------------------
    # Convert to grayscale -> Resize -> ToTensor -> Normalize
    # Typical MNIST normalization: mean=0.1307, std=0.3081
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    }

    # -------------------------------
    # 4. Load datasets and dataloaders
    # -------------------------------
    data_dir = 'data'  # Adjust to your dataset root
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x),
                                transform=data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=32,
                      shuffle=True,
                      num_workers=0)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print("Classes detected:", class_names)
    print("Training images:", dataset_sizes['train'])
    print("Validation images:", dataset_sizes['val'])

    # -------------------------------
    # 5. Create the model
    # -------------------------------
    model = SmallCNN(num_classes=len(class_names))

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -------------------------------
    # 6. Define loss and optimizer
    # -------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 50  # Increased epochs

    # -------------------------------
    # 7. Training loop
    # -------------------------------
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set to training mode
            else:
                model.eval()   # Set to eval mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backprop in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    print('Training complete')

    # -------------------------------
    # 8. Save the model
    # -------------------------------
    exporter = ModelExporter(model, class_names)
    save_path = os.path.join("saved_models", "smallcnn_28x28.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    exporter.save(save_path)

if __name__ == '__main__':
    main()

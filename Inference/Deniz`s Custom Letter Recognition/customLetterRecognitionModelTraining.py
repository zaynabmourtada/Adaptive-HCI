import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def main():
    # -------------------------------
    # 1. Define transforms for 28x28 grayscale images.
    # -------------------------------
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
    # 2. Load datasets and dataloaders.
    # -------------------------------
    data_dir = 'data'  # Change this to your dataset directory.
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print("Classes detected:", class_names)
    print("Training images:", dataset_sizes['train'])
    print("Validation images:", dataset_sizes['val'])

    # -------------------------------
    # 3. Create a ResNet‑18–based model.
    # -------------------------------
    # Load a pretrained ResNet‑18 model.
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer to accept 1-channel grayscale images.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # (Optional) Initialize conv1 weights by averaging the pretrained weights across channels:
    # model.conv1.weight.data = model.conv1.weight.data.mean(dim=1, keepdim=True)
    
    # Replace the final fully connected layer to output the correct number of classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # -------------------------------
    # 4. Set up device, loss, optimizer, and scheduler.
    # -------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 50  # Increase the number of epochs as needed

    # -------------------------------
    # 5. Training loop.
    # -------------------------------
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

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
    # 6. Save the model checkpoint.
    # -------------------------------
    save_path = os.path.join("saved_models", "resnet18_custom.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'model': model,
        'class_names': class_names
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()

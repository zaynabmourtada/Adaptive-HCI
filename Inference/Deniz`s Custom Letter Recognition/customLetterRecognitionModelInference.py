import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -------------------------------
# Define the model architecture
# -------------------------------
class SmallCNN(nn.Module):
    """
    A simple CNN for 28x28 grayscale images.
    This architecture should match the one used during training.
    """
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        # in_channels=1 because images are grayscale
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two pools: 28 -> 14 -> 7, so feature map: 64 x 7 x 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(torch.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)                   # Flatten
        x = torch.relu(self.fc1(x))                 # [batch, 128]
        x = self.fc2(x)                             # [batch, num_classes]
        return x

# -------------------------------
# Inference Engine Class
# -------------------------------
class InferenceEngine:
    def __init__(self, checkpoint_path, device):
        """
        Args:
            checkpoint_path: Path to the saved .pth model checkpoint.
            device: Device to run inference on (e.g., 'cuda:0' or 'cpu').
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model, self.class_names = self.load_checkpoint()
        # Define the transform for inference (28x28 grayscale images).
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_checkpoint(self):
        """
        Load the model checkpoint and return the model and class names.
        """
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        # Initialize the model architecture with the appropriate number of classes.
        model = SmallCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model, class_names

    def predict(self, image_path):
        """
        Perform inference on the image at image_path.
        Returns:
            predicted_letter: The letter (class) with the highest probability.
            probability: The corresponding probability.
        """
        image = Image.open(image_path)
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)  # Create a mini-batch
        
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_index = torch.max(probabilities, dim=1)
        
        predicted_letter = self.class_names[top_index.item()]
        probability = top_prob.item()
        return predicted_letter, probability

# -------------------------------
# Example Usage with Image Selection
# -------------------------------
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set the path to your saved .pth checkpoint.
    checkpoint_path = os.path.join("saved_models", "smallcnn_28x28.pth")
    if not os.path.exists(checkpoint_path):
        raise Exception(f"Checkpoint not found at {checkpoint_path}")
    
    # Instantiate the inference engine.
    inference_engine = InferenceEngine(checkpoint_path, device)
    
    # Hide the main tkinter window.
    Tk().withdraw()
    
    # Open file dialog for the user to select an image.
    image_path = askopenfilename(
        title="Select an image file for inference",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not image_path:
        raise Exception("No image file selected.")
    
    predicted_letter, probability = inference_engine.predict(image_path)
    print("Predicted Letter:", predicted_letter)
    print("Probability:", probability)
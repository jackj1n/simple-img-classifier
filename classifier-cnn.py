import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import os
from data import train_loader, test_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input channels: 3, Output channels: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Input channels: 32, Output channels: 64
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute the flattened feature size dynamically
        self._to_linear = None  # Placeholder for the flattened size
        self._compute_flattened_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)  # Use computed flattened size
        self.fc2 = nn.Linear(512, 10)  # 10 output classes

    def _compute_flattened_size(self):
        # Pass a dummy tensor through the network to calculate the flattened size
        x = torch.randn(1, 3, 32, 32)  # Dummy input: batch_size=1, channels=3, height=32, width=32
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))  # Apply convolutional and pooling layers
        self._to_linear = x.numel()  # Total number of elements in the tensor (channels * height * width)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # ReLU: max(0, x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten feature maps dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model_if_exists(model, file_path="cnn_model.pth"):
    if os.path.exists(file_path):
        print(f"Found saved model at {file_path}. Loading model...")
        model.load_state_dict(torch.load(file_path))
        print("Model loaded successfully!")
        return True
    else:
        print(f"No saved model found at {file_path}. Starting from scratch.")
        return False

def main():
    model = CNN()
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Check if a saved model exists
    if not load_model_if_exists(model):
        # Train the model if no saved model exists
        train_model(model, device)
        # Save the model after training
        torch.save(model.state_dict(), "cnn_model.pth")
        print("Model saved after training.")
    
    # Evaluate the model
    evaluate_model(model, device)

    # Visualize model
    # visualize_predictions(model, device, test_loader, train_loader.dataset.classes)

def train_model(model, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()  # Loss function: combines Softmax and NLLLoss, measures how well model predictions match true labels
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer: updates model's weights based on gradient during backprop

    # Training Loop Structure:
    # Loop through the training dataset in batches.
    # Pass each batch through the model to compute predictions.
    # Calculate the loss using the loss function.
    # Backpropagate the gradients.
    # Update the model weights using the optimizer.
    for epoch in range(num_epochs):
        running_loss = 0.0 # Keeps track of loss
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero parameter grads
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimize weights
            optimizer.step()
            
            # Update loss
            running_loss += loss.item()
        # Print average loss for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, device):
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1) # Takes class with highest score

            # Update metrics
            correct += (predicted == labels).sum().item() # Number of correct predictions
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    
def visualize_predictions(model, device, test_loader, class_names, num_images=8):
    model.eval()

    data_iter = iter(test_loader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    # Move tensors to CPU for visualization
    inputs, labels, predicted = inputs.cpu(), labels.cpu(), predicted.cpu()

    # Plot images and predictions
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        img = inputs[i] / 2 + 0.5  # Unnormalize
        img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        plt.imshow(img)
        plt.title(f"Pred: {class_names[predicted[i]]}\nTrue: {class_names[labels[i]]}")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
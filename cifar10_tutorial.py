# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================

This script trains a Convolutional Neural Network (CNN) on the CIFAR10 dataset using PyTorch.
It includes data loading, preprocessing, visualization, model definition, training, evaluation,
and plotting of loss and accuracy metrics over epochs.
Enhanced with structured logging and progress bars for better monitoring.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
import sys
import os

# Setup logging
def setup_logging():
    """
    Sets up logging to output to both console and a file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler
    if not os.path.exists('logs'):
        os.makedirs('logs')
    fh = logging.FileHandler('logs/training.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)   # Input channels: 3 (RGB), Output channels: 6, Kernel size: 5
        self.pool = nn.MaxPool2d(2, 2)    # Max pooling with kernel size 2 and stride 2
        self.conv2 = nn.Conv2d(6, 16, 5)  # Input channels: 6, Output channels: 16, Kernel size: 5

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Input features: 16*5*5, Output features: 120
        self.fc2 = nn.Linear(120, 84)          # Input features: 120, Output features: 84
        self.fc3 = nn.Linear(84, 10)           # Input features: 84, Output features: 10 (number of classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply conv1, ReLU, then pool
        x = self.pool(F.relu(self.conv2(x)))  # Apply conv2, ReLU, then pool
        x = x.view(-1, 16 * 5 * 5)            # Flatten the tensor
        x = F.relu(self.fc1(x))               # Apply fc1 and ReLU
        x = F.relu(self.fc2(x))               # Apply fc2 and ReLU
        x = self.fc3(x)                       # Apply fc3 (output layer)
        return x

# Function to display images
def imshow(img):
    """
    Displays a grid of images.
    """
    img = img / 2 + 0.5     # Unnormalize the image
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger()

    ########################################################################
    # 1. Loading and Normalizing CIFAR10
    ########################################################################

    # Define transformations for the training and testing data
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),  # Mean for each channel
                             (0.5, 0.5, 0.5))  # Standard deviation for each channel
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    logger.info("Loading CIFAR10 dataset with data augmentation for training...")
    # Download and load the training data with data augmentation
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Download and load the testing data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # Define the class labels
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    logger.info("Dataset loaded successfully.")

    ########################################################################
    # 2. Visualizing Some Training Images
    ########################################################################

    logger.info("Visualizing some training images...")
    # Get some random training images
    dataiter = iter(trainloader)
    try:
        images, labels = next(dataiter)
    except StopIteration:
        logger.error("No data available in trainloader.")
        return

    # Show images
    imshow(torchvision.utils.make_grid(images))
    logger.info('GroundTruth: ' + ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # 3. Define the Network
    ########################################################################

    logger.info("Initializing the network...")
    net = Net()
    logger.info("Network initialized.")

    ########################################################################
    # 4. Define Loss Function and Optimizer
    ########################################################################

    logger.info("Setting up the loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    logger.info("Loss function and optimizer set.")

    ########################################################################
    # 5. Training the Network
    ########################################################################

    # Determine the device to run on (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')

    # Move the network to the selected device
    net.to(device)

    num_epochs = 50  # Set to 50 as per assignment requirement

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    logger.info(f'Starting training for {num_epochs} epochs...')
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Initialize tqdm progress bar for training
        progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc='Training', ncols=100)

        for i, data in progress_bar:
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate training accuracy on the fly
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar
            if (i + 1) % 2000 == 0:
                avg_loss = running_loss / 2000
                progress_bar.set_postfix(loss=avg_loss)
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                running_loss = 0.0

        progress_bar.close()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Evaluate on test data after each epoch
        logger.info("Evaluating on test data...")
        correct = 0
        total = 0
        with torch.no_grad():
            # Initialize tqdm progress bar for testing
            test_progress = tqdm(enumerate(testloader, 0), total=len(testloader), desc='Testing', ncols=100)
            for i, data in test_progress:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move to device
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_progress.close()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        logger.info(f'Epoch {epoch + 1} Summary: Loss={epoch_loss:.3f}, Training Accuracy={epoch_accuracy:.2f}%, Test Accuracy={test_accuracy:.2f}%')

    logger.info('Finished Training')

    ########################################################################
    # 6. Testing the Network on the Test Data (Final Evaluation)
    ########################################################################

    logger.info("Final evaluation on test data...")
    # Get some random testing images
    dataiter = iter(testloader)
    try:
        images, labels = next(dataiter)
    except StopIteration:
        logger.error("No data available in testloader.")
        return

    # Show images
    imshow(torchvision.utils.make_grid(images))
    logger.info('GroundTruth: ' + ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Move images and labels to device
    images, labels = images.to(device), labels.to(device)

    # Make predictions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    logger.info('Predicted: ' + ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(4)))

    ########################################################################
    # 7. Overall Test Accuracy
    ########################################################################

    logger.info("Calculating overall test accuracy...")
    correct = 0
    total = 0
    with torch.no_grad():
        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(testloader, 0), total=len(testloader), desc='Overall Testing', ncols=100)
        for i, data in progress_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move to device
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        progress_bar.close()

    accuracy = 100 * correct / total
    logger.info(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    ########################################################################
    # 8. Per-Class Accuracy
    ########################################################################

    logger.info("Calculating per-class accuracy...")
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(testloader, 0), total=len(testloader), desc='Per-Class Testing', ncols=100)
        for i, data in progress_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move to device
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for j in range(len(labels)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
        progress_bar.close()

    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            logger.info(f'Accuracy of {classes[i]:5s} : {class_accuracy:.2f} %')
        else:
            logger.info(f'Accuracy of {classes[i]:5s} : N/A (no samples)')

    ########################################################################
    # 9. Plotting Metrics
    ########################################################################

    logger.info("Plotting training and test metrics...")
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(18, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy over Epochs')
    plt.legend()

    # Plot Test Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    ########################################################################
    # 10. GPU Utilization (Optional)
    ########################################################################

    logger.info("GPU Utilization Information:")
    if torch.cuda.is_available():
        logger.info(torch.cuda.get_device_name(0))
        logger.info(f'Allocated GPU memory: {torch.cuda.memory_allocated(0)} bytes')
        logger.info(f'Cached GPU memory: {torch.cuda.memory_reserved(0)} bytes')
    else:
        logger.info("CUDA is not available. Running on CPU.")

    ########################################################################
    # 11. Saving the Trained Model
    ########################################################################

    logger.info("Saving the trained model...")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(net.state_dict(), 'models/cifar10_net.pth')
    logger.info("Trained model saved to 'models/cifar10_net.pth'")

if __name__ == "__main__":
    main()

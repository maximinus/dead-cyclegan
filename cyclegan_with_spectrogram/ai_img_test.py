import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# This is a sample image discriminator
# We will use almost the same code in the audio discriminator
# this code is used as a test to ensure everything is correct


IMAGE_FOLDER = '/home/sparky/data/MLData/CelebA/images'
ATTR_FILE = '/home/sparky/data/MLData/CelebA/attributes.txt'
MALE_INDEX = 21
IMAGE_SIZE = 128
TEST_RATIO = 0.2
FILE_USAGE = 0.05

EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


def get_device():
    # Get the appropriate device string for PyTorch operations.
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageData(Dataset):
    def __init__(self, files):
        # Initialize the dataset by finding all WAV files in the given directory.
        self.images = []
        attributes = self.get_attributes()
        for i in files:
            value = i.split('.')[0]
            if value not in attributes:
                print(f'No file {i}')
            else:
                self.images.append([i, attributes[value]])
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()])

    def get_attributes(self):
        # open file, ignore first 2 lines
        data = {}
        with open(ATTR_FILE, 'r') as file:
            for i, line in enumerate(file):
                if i < 2:  # Skip the first two lines
                    continue
                # Process the remaining lines
                line_data = line.strip().split()
                is_male = int(line_data[MALE_INDEX])
                is_male = max(is_male, 0)
                data[line_data[0].split('.')[0]] = torch.tensor(is_male, dtype=torch.float32)
        return data

    def __getitem__(self, index):
        data = self.images[index]
        img = Image.open(f'{IMAGE_FOLDER}/{data[0]}')
        img = self.transform(img)
        return img, data[1]

    def __len__(self):
        # Get the total number of items in the dataset.
        return len(self.images)


def get_dataloaders():
    all_files = os.listdir(IMAGE_FOLDER)
    random.shuffle(all_files)
    all_files = all_files[:int(len(all_files) * FILE_USAGE)]

    print(f'* Using a total of {len(all_files)} files')

    split_index = int(len(all_files) * TEST_RATIO)

    # Split the list
    test = all_files[:split_index]
    train = all_files[split_index:]

    train_data = DataLoader(ImageData(train), batch_size=BATCH_SIZE, shuffle=True)
    test_data = DataLoader(ImageData(test), batch_size=BATCH_SIZE, shuffle=False)

    return train_data, test_data


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        # The size of the input features to the first fully connected layer is 16*16*128.
        # This is because the original image size is 128x128, and it goes through three
        # convolution layers and three pooling layers, each reducing the size by half.
        self.fc1 = nn.Linear(16 * 16 * 128, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 16 * 16 * 128)

        # Apply fully connected layers with ReLU
        x = F.relu(self.fc1(x))

        # Output layer with a single neuron for binary classification
        x = torch.sigmoid(self.fc2(x))

        return x


def training_loop(train_data, test_data):
    device = get_device()

    model = ImageClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    results = []

    for epoch in range(EPOCHS):
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for images, labels in tqdm(train_data):
            # Move tensors to the configured device
            images = images.to(device)
            # Reshape labels for BCELoss
            labels = labels.to(device).float().view(-1, 1)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep the loss computed
            train_loss += loss.item()
            predicted = outputs.data > 0.5
            correct_predictions += (predicted == labels.float().view(-1, 1)).sum().item()
            total_predictions += labels.size(0)

        train_loss /= len(train_data)
        train_acc = 100 * correct_predictions / total_predictions

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            for images, labels in test_data:
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)

                outputs = model(images)
                loss = criterion(outputs, labels.float().view(-1, 1))
                predicted = (outputs.data > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += loss.item()

            test_loss /= len(test_data)
            test_acc = 100 * correct / total
            results.append([train_loss, train_acc, test_loss, test_acc])

            train_result = f'Loss: {train_loss:.2f}, Acc: {train_acc:.2f}%'
            test_result = f'Test loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}%'
            print(f'Epoch #{epoch}: {train_result}, {test_result}')

    plot_graphs(results)


def plot_graphs(metrics):
        """
        Plot training and testing loss and accuracy.

        Args:
        metrics (list of lists): Each element is a list of
                                 [train_loss, train_accuracy, test_loss, test_accuracy]
        """
        # Unpack metrics
        train_loss = [item[0] for item in metrics]
        train_accuracy = [item[1] for item in metrics]
        test_loss = [item[2] for item in metrics]
        test_accuracy = [item[3] for item in metrics]

        # Plotting
        epochs = [x for x in range(1, len(metrics) + 1)]
        plt.figure(figsize=(10, 5))

        # Plot training and testing loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, color='blue', label='Training Loss')
        plt.plot(epochs, test_loss, color='red', label='Testing Loss')
        plt.title('Training and Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and testing accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, color='green', label='Training Accuracy')
        plt.plot(epochs, test_accuracy, color='orange', label='Testing Accuracy')
        plt.title('Training and Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    train, test = get_dataloaders()
    #images, labels = next(iter(train))
    #show_data(images, labels)
    training_loop(train, test)

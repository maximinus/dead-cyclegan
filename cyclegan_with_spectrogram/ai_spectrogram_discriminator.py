import datetime
import os
import random
import sys

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

SPECTROGRAM_FOLDER = Path('/home/sparky/data/MLData/GD_MEL')
IMAGE_OUTPUT = '/home/sparky/data/code/dead-cyclegan/output/images'
AUDIO_SIZE = 65536
TEST_RATIO = 0.2
FILE_USAGE = 0.1

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


def get_device():
    # Get the appropriate device string for PyTorch operations.
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageData(Dataset):
    def __init__(self, files):
        # Initialize the dataset by finding all WAV files in the given directory.
        self.files = []
        for i in tqdm(files):
            filepath = f'{SPECTROGRAM_FOLDER}/{i}'
            # this gives input data in the range -1 -> +1
            # HOWEVER, we need input data in the range 0-1, because that's the range the generators work in
            # we can convert the range to be correct when we output as a wav file
            waveform, sample_rate = torchaudio.load(filepath)
            waveform += 1.0
            waveform /= 2.0
            self.files.append([i, waveform])

    def __getitem__(self, index):
        label = 1.0 if 'SBD' in self.files[index][0] else 0.0
        return self.files[index][1], torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        # Get the total number of items in the dataset.
        return len(self.files)


def get_dataloaders():
    all_files = os.listdir(SPECTROGRAM_FOLDER)
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


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=4, padding=0)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=0)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=4, stride=4, padding=0)

        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Compute the flattened size after Conv and Pooling layers
        self.flattened_size = (65536 // 8 // 8 // 8 // 8) * 256

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Forward pass through Convolutional layers with ReLU and Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, self.flattened_size)

        # Apply dropout
        x = self.dropout(x)

        # Forward pass through the fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer with sigmoid activation
        x = torch.sigmoid(self.fc3(x))
        return x


class ClippedLinearActivation(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-1, max=1)


class AbsoluteMaxPooling1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AbsoluteMaxPooling1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        # Unfold the tensor into sliding local blocks (1D)
        unfolded = F.unfold(x.unsqueeze(1), kernel_size=(1, self.kernel_size), stride=(1, self.stride),
                            padding=(0, self.padding))

        # Reshape and take the absolute value
        unfolded = unfolded.view(x.size(0), x.size(1), -1, self.kernel_size)
        unfolded = torch.abs(unfolded)

        # Perform max pooling on the absolute values
        pooled, _ = unfolded.max(dim=-1)
        return pooled.squeeze(1)


def training_loop(train_data, test_data):
    device = get_device()

    # AudioClassifierAlt is not that far off, but this is the "true" way
    model = AudioClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    results = []

    for epoch in range(EPOCHS):
        model.train()
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

        # Format the date and time as a string
        datestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M")
        # Example: Create a filename for a .png file
        filepath = f'{IMAGE_OUTPUT}/{datestamp}_E{EPOCHS}_FU{int(FILE_USAGE * 100)}.png'
        plt.savefig(filepath)
        plt.show()


def free_cuda_memory():
    torch.cuda.empty_cache()


def test_model_output():
    model = AudioClassifier()
    dummy_input = torch.randn(1, 1, 65536)  # Dummy input for demonstration
    output = model(dummy_input)
    print(output.shape)
    sys.exit()


if __name__ == '__main__':
    #test_model_output()
    free_cuda_memory()
    train, test = get_dataloaders()
    training_loop(train, test)

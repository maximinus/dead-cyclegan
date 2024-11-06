import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


ROOT_FOLDER = '/home/sparky/data/MLData/GD_sliced'
TOTAL_EPOCHS = 3
BATCH_SIZE = 32
EPOCH_DISPLAY = 2
TEST_RATIO = 0.2
FILE_USAGE = 0.1
LEARNING_RATE = 0.0001


def get_device():
    # Get the appropriate device string for PyTorch operations.
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioData(Dataset):
    def __init__(self, files):
        # Initialize the dataset by finding all WAV files in the given directory.
        self.audio = []
        for i in tqdm(files):
            waveform, sample_rate = torchaudio.load(i)
            label = 1.0 if 'SBD' in i else 0.0
            self.audio.append([waveform, torch.tensor(label, dtype=torch.float32)])

    def __getitem__(self, index):
        data = self.audio[index]
        return data[0], data[1]

    def __len__(self):
        # Get the total number of items in the dataset.
        return len(self.audio)


def find_directories_with_wavs(directory):
    directories_with_wavs = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                if root not in directories_with_wavs:
                    directories_with_wavs.append(root)
                # Stop checking other files in this directory
                break
    return directories_with_wavs


def get_dataloaders():
    all_dirs = find_directories_with_wavs(ROOT_FOLDER)
    file_paths = []
    for wav_dir in all_dirs:
        for file in os.listdir(wav_dir):
            if file.endswith('.wav'):
                file_paths.append(os.path.join(wav_dir, file))

    random.shuffle(file_paths)
    file_paths = file_paths[:int(len(file_paths) * FILE_USAGE)]

    print(f'* Using a total of {len(file_paths)} files')

    split_index = int(len(file_paths) * TEST_RATIO)

    # Split the list
    test = file_paths[:split_index]
    train = file_paths[split_index:]

    train_data = DataLoader(AudioData(train), batch_size=BATCH_SIZE, shuffle=True)
    test_data = DataLoader(AudioData(test), batch_size=BATCH_SIZE, shuffle=False)

    return train_data, test_data


class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()

        # Define the 1D convolutional layers with batch normalization, ReLU, and dropout
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        # Adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Define the output linear layer
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))

        # Adaptive average pooling
        x = self.avg_pool(x)

        # Flatten the tensor for the linear layer
        x = x.view(x.size(0), -1)

        # Forward pass through the output linear layer
        x = self.fc(x)
        return x


def calculate_accuracy(logits, targets):
    probs = torch.sigmoid(logits)
    predicted = (probs >= 0.5).float()
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)


def train_model(model, train_data, test_data, loss_function, optimizer, device):
    """
    Train the PyTorch model.

    Args:
    model (torch.nn.Module): The neural network model to train.
    dataloader (torch.utils.data.DataLoader): The DataLoader for the training data.
    loss_function (torch.nn.modules.loss): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    epochs (int): The number of epochs to train for.
    """
    model.train()  # Set the model to training mode

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        for data, target in tqdm(train_data, desc=f'Epoch {epoch+1}/{TOTAL_EPOCHS}'):
            # Move data and target to the same device as the model
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            target = target.unsqueeze(1)
            loss = loss_function(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_correct += calculate_accuracy(output, target) * target.size(0)
            train_total += target.size(0)

        train_accuracy = 100.0 * train_correct / train_total

        model.eval()  # Set the model to evaluation mode

        test_correct = 0
        test_total = 0
        total_loss = 0
        with torch.no_grad():  # No need to track gradients during evaluation
            for batch in test_data:
                data, target = batch
                data, target = data.to(device), target.to(device)

                output = model(data)
                target = target.unsqueeze(1)
                loss = loss_function(output, target)
                total_loss += loss.item()

                test_correct += calculate_accuracy(output, target) * target.size(0)
                test_total += target.size(0)

        test_accuracy = 100 * test_correct / test_total
        avg_loss = total_loss / len(test_data)

        if epoch > 1 and epoch % EPOCH_DISPLAY == 0:
            print(f'\nEpoch [{epoch + 1}/{TOTAL_EPOCHS}] TL: {loss.item():.3f}, TA: {train_accuracy:.3f}%, TeL: {avg_loss:.3f}, TeA: {test_accuracy:.3f}%')


def test_model():
    model = AudioNet()
    example_input = torch.randn(32, 1, 16384)  # Random input tensor of shape [batch, channels, length]
    output = model(example_input)
    print(example_input.shape)
    print(output.shape)  # Should be torch.Size([1, 1])


def show_dataloader_sizes(dataloader):
    for data, target in dataloader:
        print("Data batch shape:", data.shape)
        print("Target batch shape:", target.shape)
        # After checking the first batch, we exit the loop
        break


if __name__ == '__main__':
    device = get_device()
    print(f'* Using {device}')
    model = AudioNet().to(device)
    train, test = get_dataloaders()

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()
    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, train, test, loss_function, optimizer, device)

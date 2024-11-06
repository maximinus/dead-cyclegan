import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gd_data.loaders import get_test_train, get_all_wav_files

EPOCHS = 50
EPOCH_RESULTS_EVERY = 5
RAND_SEED = 5
FILE_TOTAL = 0.3
TEST_RATIO = 0.2
LEARNING_RATE = 0.000001


def get_device():
    if torch.cuda.is_available():
        print('* Using GPU')
        return 'cuda'
    else:
        print('* Using CPU')
    return 'cpu'


class Discriminator(nn.Module):
    """
    We should:
        1: Use strided convolutions
        2: Use batch normalization
        3: Use ReLU
        4: Use Adam optimization
        5: Use tanh as the activation function?
            It's fine, as the ReLU already clears all scores <0
             If we do this, the return labels could be negative
        6: Use BCELoss
    """
    def __init__(self):
        super().__init__()
        # the input size is (batch, 1, 16384)
        # (1, 16384) -> (32, 4096)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # (32, 4096) -> (64, 1024)
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # (64, 1024) -> (128, 256)
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128*256, out_features=1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activate(x)
        return x


def calculate_accuracy(base_truth, prediction):
    total_correct = float(torch.eq(base_truth, prediction).sum().item())
    return (total_correct / len(base_truth)) * 100.0


def train_network():
    torch.manual_seed(RAND_SEED)
    device = get_device()

    train_data, test_data = get_test_train(file_usage=FILE_TOTAL, test_ratio=TEST_RATIO)
    model = Discriminator().to(device)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0
        for x, y in train_data:
            x = x.to(device)
            y = y.float()
            y = y.to(device)
            y_pred = model(x).squeeze()

            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_data)

        model.eval()
        test_loss = 0
        with torch.inference_mode():
            for x_test, y_test in test_data:
                x_test = x_test.to(device)
                y_test = y_test.float()
                y_test = y_test.to(device)
                y_pred = model(x_test).squeeze()

                test_loss += loss_fn(y_pred, y_test)
            test_loss /= len(test_data)

        if epoch > 0 and epoch % EPOCH_RESULTS_EVERY == 0:
            print(f'Epoch: {epoch}:\tLoss: {loss:.3f}\tTest loss: {test_loss:.3f}')


if __name__ == '__main__':
    train_network()

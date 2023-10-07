import os
import random
import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.000003
RANDOM_SEED = 5

# ratio of files to train against
RATIO = 0.85
# what ratio of files to actuaLLY use
USE_FILES = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU')
else:
    print('Using CPU')


#ROOT_FOLDER = Path.cwd() / 'data' / 'Linear'
ROOT_FOLDER = Path.cwd() / 'data' / 'Split'


class AudioData(Dataset):
    def __init__(self, files):
        super().__init__()
        self.all_files = [(x, 1) if x.startswith('SBD') else (x, 0) for x in files]

    def __getitem__(self, index):
        data = self.all_files[index]
        sfx = np.load(ROOT_FOLDER / data[0])
        # return a tensor, not an array
        # this gives a 131072 Tensor, but we need (1, 131072) shape
        audio = torch.from_numpy(sfx)
        audio = (audio[None, :]).float()
        return audio, data[1]

    def __len__(self):
        return len(self.all_files)


def get_audio_loaders():
    files = os.listdir(ROOT_FOLDER)

    #sbd_files = []
    #aud_files = []
    #for i in files:
    #    if i.startswith('SBD'):
    #        sbd_files.append(i)
    #    else:
    #        aud_files.append(i)

    random.shuffle(files)
    files_to_use = int(len(files) * USE_FILES)
    files = files[:files_to_use]

    test_length = int(RATIO * len(files))

    train_data = AudioData(files[test_length:])
    test_data = AudioData(files[:test_length])

    print(f'Training Files: {len(train_data)}')
    print(f'    Test files: {len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return test_loader, train_loader


def get_model():
    model = nn.Sequential()

    LAYER1 = 200
    LAYER2 = 50
    LAYER3 = 40
    LAYER4 = 40

    model.add_module('conv1', nn.Conv1d(in_channels=1, out_channels=LAYER1, kernel_size=16, stride=16))
    model.add_module('norm1', nn.BatchNorm1d(LAYER1))
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool1d(kernel_size=2))
    #model.add_module('dropout1', nn.Dropout(p=0.5))

    #model.add_module('conv2', nn.Conv1d(in_channels=LAYER1, out_channels=LAYER2, kernel_size=10, stride=10))
    #model.add_module('norm2', nn.BatchNorm1d(LAYER2))
    #model.add_module('relu2', nn.ReLU())
    #model.add_module('pool2', nn.MaxPool1d(kernel_size=2))
    #model.add_module('dropout2', nn.Dropout(p=0.5))

    #model.add_module('conv3', nn.Conv1d(in_channels=LAYER2, out_channels=LAYER3, kernel_size=6, stride=6))
    #model.add_module('norm3', nn.BatchNorm1d(LAYER3))
    #model.add_module('relu3', nn.ReLU())
    #model.add_module('pool3', nn.MaxPool1d(kernel_size=2))
    #model.add_module('dropout3', nn.Dropout(p=0.5))

    #model.add_module('conv4', nn.Conv1d(in_channels=LAYER3, out_channels=LAYER4, kernel_size=4, stride=4))
    #model.add_module('norm4', nn.BatchNorm1d(LAYER4))

    model.add_module('pool6', nn.AvgPool1d(kernel_size=24))
    model.add_module('flatten1', nn.Flatten())

    model.add_module('fc1', nn.Linear(34000, 1))
    #model.add_module('fc2', nn.Linear(32, 1))
    model.add_module('sigmoid1', nn.Sigmoid())
    model = model.to(device)
    return model


def train(model, train_dl, valid_dl):
    loss_hist_train = [0] * EPOCHS
    accuracy_hist_train = [0] * EPOCHS
    loss_hist_valid = [0] * EPOCHS
    accuracy_hist_valid = [0] * EPOCHS

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch in tqdm(train_dl):
            x_batch = x_batch.to(device)
            # ybatch is a tensor of 0's and 1's
            y_batch = y_batch.to(device)
            # Note the result is a tensor of shape (32, ), done with the [:, 0] at the end
            # [:, 0] means, for all data, select column 0. 32 is the batch number
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # loss.item is the python value of the tensor
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in tqdm(valid_dl):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


def draw_graph(history):
    x_arr = np.arange(len(history[0])) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, history[0], '-o', label='Train loss')
    ax.plot(x_arr, history[1], '--<', label='Validation loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, history[2], '-o', label='Train acc.')
    ax.plot(x_arr, history[3], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)

    # plt.savefig('figures/14_17.png', dpi=300)
    plt.show()


def test_accuracy(model, test_dl):
    accuracy_test = 0
    model.eval()
    print('Validating')
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_dl):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)[:, 0]
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            accuracy_test += is_correct.sum().cpu()

    accuracy_test /= len(test_dl.dataset)
    print(f'Test accuracy: {accuracy_test:.4f}')


def save_model(model):
    path = 'models/celeba-cnn.ph'
    torch.save(model, path)


def show_model():
    model = get_model()
    summary(model, (1, 131072))
    sys.exit()


def build_model():
    show_model()
    train_data, test_data = get_audio_loaders()

    neural_model = get_model()
    model_history = train(neural_model, train_data, test_data)
    test_accuracy(neural_model, test_data)
    draw_graph(model_history)


if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    build_model()

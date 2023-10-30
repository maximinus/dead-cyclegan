import os
import sys
import time
import shutil
import random

import librosa
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Best guide so far:
# https://sofiadutta.github.io/datascience-ipynbs/pytorch/CycleGAN_Img_Translation_PyTorch_Horse2Zebra.html

# Current questions:
# Why are the loss values so different?
# What should they be?
# What denotes success here?

# Next work:
# Automate saving the graph image
# Play with hyperparameters to achieve success
# Try converting some real audio
# When successful, can we try non-normalised data?

AUDIO_SOURCE_DIR = Path('/home/sparky/code/dead-cyclegan/data/split')

BATCH_SIZE = 32
EPOCHS = 120
# epoch in which to slow down the learning rate
DECAY_EPOCH = 150
EPOCH_OFFSET = 1
RANDOM_SEED = 5

# learning rates
GEN_LEARNING_RATE = 0.0001
DIS_LEARNING_RATE = 0.000005

# ratio of files to train against
RATIO = 0.95
# what ratio of files to actually use
USE_FILES = 0.3

# resnet layers to add - suggested is 9 (!)
TOTAL_RESNETS = 6
RESNET_CHANNELS = 512

# size of layers
LAYER1 = 64
LAYER2 = 128
LAYER3 = 512

REAL_LABEL = 1.0
FAKE_LABEL = 0.0

SAMPLE_RATE = 44100


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('* Using GPU')
else:
    print('* Using CPU')


ROOT_FOLDER = Path.cwd() / 'data' / 'split'
EXAMPLES_FOLDER = Path.cwd() / 'examples'
MODEL_DIRECTORY = Path.cwd() / 'models'


class AudioData(Dataset):
    def __init__(self, files):
        super().__init__()
        self.all_files = files
        # load all files into RAM!
        self.audio = []
        for i in tqdm(self.all_files):
            self.audio.append(self.load_file(i))

    def show_top(self, total):
        files = [self.all_files[x] for x in range(total)]
        print(f'Top #{total}: {files}')

    def load_file(self, filename):
        signal, audio = librosa.core.load(filename, sr=SAMPLE_RATE, mono=False)
        signal = signal.astype(np.float32)
        signal /= 32768.0

        # reshape the signal
        n_channels = 1 if signal.ndim == 1 else signal.shape[1]
        signal = signal.reshape(n_channels, signal.shape[0])

        # normalize the signal
        signal /= np.max(np.abs(signal))
        return signal

    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return len(self.all_files)


def get_all_wav_files(audio_dir):
    wav_files = []
    for song_dir in os.listdir(audio_dir):
        full_path = audio_dir / song_dir
        if os.path.isdir(full_path):
            for possible_wav in os.listdir(full_path):
                if str(possible_wav).lower().endswith('wav'):
                    wav_files.append(full_path / possible_wav)
    return wav_files


def get_fileset(string_end):
    show_dirs = os.listdir(AUDIO_SOURCE_DIR)
    audio_samples = []
    for dir in show_dirs:
        # look for folders ending
        if os.path.isdir(AUDIO_SOURCE_DIR / dir) and str(dir.endswith(string_end)):
            audio_samples.extend(get_all_wav_files(AUDIO_SOURCE_DIR / dir))
    return audio_samples


def get_audio_loaders():

    sbd_files = get_fileset('SBD')
    aud_files = get_fileset('AUD')

    print(f'* Found {len(sbd_files)} SBD files')
    print(f'* Found {len(aud_files)} AUD files')

    random.shuffle(sbd_files)
    random.shuffle(aud_files)

    sbd_files_to_use = int(len(sbd_files) * USE_FILES)
    sbd_files = sbd_files[:sbd_files_to_use]
    aud_files_to_use = int(len(aud_files) * USE_FILES)
    aud_files = aud_files[:aud_files_to_use]

    print(f'* Using {len(sbd_files)} SBD files')
    print(f'* Using {len(sbd_files)} AUD files')

    sbd_test_length = int(RATIO * len(sbd_files))
    aud_test_length = int(RATIO * len(aud_files))

    train_sbd_data = AudioData(sbd_files[:sbd_test_length])
    train_aud_data = AudioData(aud_files[:aud_test_length])
    test_sbd_data = AudioData(sbd_files[sbd_test_length:])
    test_aud_data = AudioData(aud_files[aud_test_length:])

    train_sbd_loader = DataLoader(train_sbd_data, batch_size=BATCH_SIZE, shuffle=True)
    train_aud_loader = DataLoader(train_aud_data, batch_size=BATCH_SIZE, shuffle=True)
    test_sbd_loader = DataLoader(test_sbd_data, batch_size=BATCH_SIZE)
    test_aud_loader = DataLoader(test_aud_data, batch_size=BATCH_SIZE)

    # the length of the loaders must be multiplied by the
    print(f'* SBD Train: {len(train_sbd_loader) * BATCH_SIZE} files')
    print(f'* AUD Train: {len(train_aud_loader) * BATCH_SIZE} files')
    print(f'*  SBD Test: {len(test_sbd_loader) * BATCH_SIZE} files')
    print(f'*  AUD Test: {len(test_aud_loader)* BATCH_SIZE} files')

    return train_sbd_loader, train_aud_loader, test_sbd_loader, test_aud_loader


class AudioHistoryBuffer:
    def __init__(self):
        # store a collection of previously generated audio of each domain
        # this is used to update the discriminator models
        # add audio to buffer until full (50)
        # after that either add audio (50% chance) or use a generated fake directly
        # the history buffer helps the discriminator not to forget what it has done wrong before
        self.buffer = []

    def update_buffer(self, new_audio):
        return_audio = []
        for input_audio in new_audio.data:
            input_audio = torch.stack([input_audio])
            if len(self.buffer) < 50:
                self.buffer.append(input_audio)
                return_audio.append(input_audio)
            elif random.random() > 0.5:
                index = random.randint(0, 49)
                return_audio.append(self.buffer[index])
                self.buffer[index] = input_audio
            else:
                return_audio.append(input_audio)
        return torch.cat(return_audio, 0)


def get_discriminator():
    model = nn.Sequential()

    model.add_module('conv1', nn.Conv1d(in_channels=1, out_channels=LAYER1, kernel_size=16, stride=16))
    model.add_module('norm1', nn.BatchNorm1d(LAYER1))
    model.add_module('relu1', nn.ReLU())

    model.add_module('conv2', nn.Conv1d(in_channels=LAYER1, out_channels=LAYER2, kernel_size=8, stride=8))
    model.add_module('norm2', nn.BatchNorm1d(LAYER2))
    model.add_module('relu2', nn.ReLU())

    model.add_module('conv3', nn.Conv1d(in_channels=LAYER2, out_channels=LAYER3, kernel_size=8, stride=8))
    model.add_module('norm3', nn.BatchNorm1d(LAYER3))
    model.add_module('relu3', nn.ReLU())

    model.add_module('conv4', nn.Conv1d(in_channels=LAYER3, out_channels=1, kernel_size=16, stride=16))
    model.add_module('sigmoid1', nn.Sigmoid())
    model = model.to(device)
    return model


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # Batch size is 1, so use instancenorm
        # input -> Conv -> InstanceNorm -> Relu -> Conv -> InstanceNorm
        self.conv = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm1d(channels),
                                  nn.ReLU(),
                                  nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm1d(channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        # add the original input to the output
        out += residual
        # and a final ReLU after addition - seen to give best results
        return self.relu(out)


def get_generator():
    model = nn.Sequential()

    model.add_module('conv1', nn.Conv1d(in_channels=1, out_channels=LAYER1, kernel_size=16, stride=16))
    model.add_module('norm1', nn.BatchNorm1d(LAYER1))
    model.add_module('relu1', nn.ReLU())

    model.add_module('conv2', nn.Conv1d(in_channels=LAYER1, out_channels=LAYER2, kernel_size=8, stride=8))
    model.add_module('norm2', nn.BatchNorm1d(LAYER2))
    model.add_module('relu2', nn.ReLU())

    model.add_module('conv3', nn.Conv1d(in_channels=LAYER2, out_channels=LAYER3, kernel_size=8, stride=8))
    model.add_module('norm3', nn.BatchNorm1d(LAYER3))
    model.add_module('relu3', nn.ReLU())

    # resnet blocks go here (sometimes called residual blocks)
    resnet_index = 1
    for i in range(TOTAL_RESNETS):
        model.add_module(f'resnet{resnet_index}', ResidualBlock(RESNET_CHANNELS))
        resnet_index += 1

    model.add_module('norm4', nn.BatchNorm1d(LAYER3))
    model.add_module('relu4', nn.ReLU())
    model.add_module('iconv1', nn.ConvTranspose1d(in_channels=LAYER3, out_channels=LAYER2, kernel_size=8, stride=8))

    model.add_module('norm5', nn.BatchNorm1d(LAYER2))
    model.add_module('relu5', nn.ReLU())
    model.add_module('iconv2', nn.ConvTranspose1d(in_channels=LAYER2, out_channels=LAYER1, kernel_size=8, stride=8))

    model.add_module('norm6', nn.BatchNorm1d(LAYER1))
    model.add_module('relu6', nn.ReLU())
    model.add_module('iconv3', nn.ConvTranspose1d(in_channels=LAYER1, out_channels=1, kernel_size=16, stride=16))

    model.add_module('tanh1', nn.Tanh())

    model = model.to(device)
    return model


class StatsTracker:
    def __init__(self):
        self.disc_sbd_losses = []
        self.disc_aud_losses = []
        self.gen_sbd_to_aud_losses = []
        self.gen_aud_to_sbd_losses = []
        self.cycle_sbd_losses = []
        self.cycle_aud_losses = []

    def draw_graph(self):
        plt.figure(figsize=(10, 5))
        plt.title('Generators / Discriminators losses and cyclic losses', fontsize=14)
        plt.xlabel('Number of Epochs', fontsize=14)
        plt.ylabel('Train Losses', fontsize=14)

        plt.plot(self.disc_sbd_losses, label='Disc SBD losses')
        plt.plot(self.disc_aud_losses, label='Disc AUD losses')
        plt.plot(self.gen_sbd_to_aud_losses, label='SBD to AUD losses')
        plt.plot(self.gen_aud_to_sbd_losses, label='AUD to SBD losses')
        plt.plot(self.cycle_sbd_losses, label='Cycle SBD losses')
        plt.plot(self.cycle_aud_losses, label='Cycle AUD losses')

        plt.legend()
        plt.show()


def save_model(aud_to_sbd_model):
    # filename is YY_MM_DD_HH.pth
    now = datetime.now()
    filename = f'{str(now.year)[:2]}_{now.month:02d}_{now.day:02d}_{now.hour:02d}.pth'
    filepath = MODEL_DIRECTORY / filename
    torch.save(aud_to_sbd_model.state_dict(), filepath)
    print(f'* Saved model to {filepath}')


def weights_init(m):
    # Weight initialization from a Gaussian distribution N (0, 0.02)
    for layer in m.children():
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.ConvTranspose1d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def train_cyclegan():
    # get data
    train_sbd, train_aud, test_sbd, test_aud = get_audio_loaders()

    disc_sbd = get_discriminator()
    disc_aud = get_discriminator()
    disc_sbd.apply(weights_init)
    disc_aud.apply(weights_init)
    print('* Created discriminators')

    gen_aud_to_sbd = get_generator()
    gen_sbd_to_aud = get_generator()
    gen_aud_to_sbd.apply(weights_init)
    gen_sbd_to_aud.apply(weights_init)
    print('* Created generators')

    generators_parameters = list(gen_aud_to_sbd.parameters()) + list(gen_sbd_to_aud.parameters())
    optimizer_gen = torch.optim.AdamW(generators_parameters, lr=GEN_LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_disc_sbd = torch.optim.AdamW(disc_sbd.parameters(), lr=DIS_LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_disc_aud = torch.optim.AdamW(disc_aud.parameters(), lr=DIS_LEARNING_RATE, betas=(0.5, 0.999))

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    print('* Created optimizers and loss functions')

    stats = StatsTracker()

    # learning rate update schedulers
    #lambda_lr_func = lambda epoch: 1.0 - max(0, epoch + EPOCH_OFFSET - DECAY_EPOCH) / (EPOCHS - DECAY_EPOCH)
    #lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lambda_lr_func)
    #lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_disc_sbd, lr_lambda=lambda_lr_func)
    #lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_disc_aud, lr_lambda=lambda_lr_func)

    sbd_buffer = AudioHistoryBuffer()
    aud_buffer = AudioHistoryBuffer()

    for epoch in range(EPOCHS):
        # time how long
        start = time.time()

        # put models in training mode
        gen_aud_to_sbd = gen_aud_to_sbd.train()
        gen_sbd_to_aud = gen_sbd_to_aud.train()
        disc_sbd = disc_sbd.train()
        disc_aud = disc_aud.train()

        g_sbd_to_aud_running_loss = 0.0
        g_aud_to_sbd_running_loss = 0.0
        cycle_x_running_loss = 0.0
        cycle_y_running_loss = 0.0
        d_sbd_running_loss = 0.0
        d_aud_running_loss = 0.0

        print(f'* Starting Epoch {epoch+1}: Approx {len(train_sbd)} iters')

        for real_sbd, real_aud in tqdm(zip(train_sbd, train_aud)):

            real_sbd = real_sbd.to(device)
            real_aud = real_aud.to(device)

            # train the generators
            optimizer_gen.zero_grad()

            # take the real sbd and convert to a fake aud
            fake_aud = gen_sbd_to_aud(real_sbd)
            d_a_fake_out = disc_aud(fake_aud)
            gen_sbd_to_aud_loss = mse_criterion(d_a_fake_out, torch.full(d_a_fake_out.size(), REAL_LABEL, device=device))
            reconstructed_sbd = gen_aud_to_sbd(fake_aud)
            cycle_x_loss = l1_criterion(reconstructed_sbd, real_sbd) * 10.0

            # take the real aud and convert to a fake sbd
            fake_sbd = gen_aud_to_sbd(real_aud)
            d_s_fake_out = disc_sbd(fake_sbd)
            gen_aud_to_sbd_loss = mse_criterion(d_s_fake_out, torch.full(d_s_fake_out.size(), REAL_LABEL, device=device))
            reconstructed_aud = gen_sbd_to_aud(fake_sbd)
            # lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation
            cycle_y_loss = l1_criterion(reconstructed_aud, real_aud) * 10.0

            # total all loss and back propagate
            g_loss = gen_sbd_to_aud_loss + gen_aud_to_sbd_loss + cycle_x_loss + cycle_y_loss
            g_loss.backward()

            # update all parameters
            optimizer_gen.step()

            g_aud_to_sbd_running_loss += gen_aud_to_sbd_loss.item()
            g_sbd_to_aud_running_loss += gen_sbd_to_aud_loss.item()
            cycle_x_running_loss += cycle_x_loss.item()
            cycle_y_running_loss += cycle_y_loss.item()

            # now we train the discriminators
            # -------------------------------
            optimizer_disc_sbd.zero_grad()
            # train with real sbd
            d_s_real_out = disc_sbd(real_sbd)
            d_s_real_loss = mse_criterion(d_s_real_out, torch.full(d_s_real_out.size(), REAL_LABEL, device=device))

            # train with fake sbd
            fake_sbd = sbd_buffer.update_buffer(fake_sbd)

            # compute the fake loss for discriminator on fake audio generated by generator
            d_s_fake_out = disc_sbd(fake_sbd)
            d_s_fake_loss = mse_criterion(d_s_fake_out, torch.full(d_s_fake_out.size(), FAKE_LABEL, device=device))

            # * 0.5 to slow down update, else it will learn too fast
            d_sbd_loss = (d_s_real_loss + d_s_fake_loss) * 0.5
            d_sbd_loss.backward()
            optimizer_disc_sbd.step()
            d_sbd_running_loss += d_sbd_loss.item()

            # then repeat for the second discriminator
            optimizer_disc_aud.zero_grad()
            d_a_real_out = disc_aud(real_aud)
            d_a_real_loss = mse_criterion(d_a_real_out, torch.full(d_a_real_out.size(), REAL_LABEL, device=device))
            fake_aud = aud_buffer.update_buffer(fake_aud)
            d_a_fake_out = disc_aud(fake_aud)
            d_a_fake_loss = mse_criterion(d_a_fake_out, torch.full(d_a_fake_out.size(), FAKE_LABEL, device=device))
            d_aud_loss = (d_a_real_loss + d_a_fake_loss) * 0.5
            d_aud_loss.backward()
            optimizer_disc_aud.step()
            d_aud_running_loss += d_aud_loss.item()


        # epoch has finished
        end = time.time()
        total_time = end - start

        print(f'* Epoch #{epoch}')
        print(f'*   SBD loss: {d_sbd_running_loss:.2f}, AUD loss: {d_aud_running_loss:.2f}')
        print(f'*   Sbd to Aud loss: {g_sbd_to_aud_running_loss:.2f}')
        print(f'*   Aud to Sbd loss: {g_aud_to_sbd_running_loss:.2f}')
        print(f'*   Cycle sbd loss: {cycle_x_running_loss}')
        print(f'*   Cycle aud loss: {cycle_y_running_loss}')
        print(f'* Took {total_time:.2f}s')
        print('----------')

        stats.disc_aud_losses.append(d_aud_running_loss)
        stats.disc_sbd_losses.append(d_sbd_running_loss)
        stats.gen_sbd_to_aud_losses.append(g_sbd_to_aud_running_loss)
        stats.gen_aud_to_sbd_losses.append(g_aud_to_sbd_running_loss)
        stats.cycle_sbd_losses.append(cycle_x_running_loss)
        stats.cycle_aud_losses.append(cycle_y_running_loss)

        # update learning rates
        #lr_scheduler_G.step()
        #lr_scheduler_D_X.step()
        #lr_scheduler_D_Y.step()

    save_model(gen_aud_to_sbd)
    stats.draw_graph()


def show_model(generator=False):
    print('Generator Shape:')
    if generator is True:
        model = get_generator()
    else:
        print('Discriminator Shape:')
        model = get_discriminator()
    summary(model, (1, 16384))
    sys.exit()


def show_residual():
    print('Residual Shape:')
    res = ResidualBlock(RESNET_CHANNELS)
    res = res.to(device)
    summary(res, (RESNET_CHANNELS, 16))
    sys.exit()


def clear_examples():
    shutil.rmtree(EXAMPLES_FOLDER)
    os.mkdir(EXAMPLES_FOLDER)


def build_model():
    # show_model(generator=True)
    # show_residual()
    clear_examples()
    train_cyclegan()


if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    build_model()

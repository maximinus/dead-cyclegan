import os
import random
from datetime import datetime

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import optim


AUDIO_FOLDER = '/home/sparky/data/MLData/GD_sliced'
AUDIO_OUTPUT = '/home/sparky/data/code/dead-cyclegan/output/audio'
IMAGE_OUTPUT = '/home/sparky/data/code/dead-cyclegan/output/images'

TEST_RATIO = 0.1
FILE_USAGE = 0.2

EPOCHS = 8
BATCH_SIZE = 8
LEARNING_RATE = 0.00005
DISCRIMINATOR_SLOWDOWN = 8
TOTAL_RESNETS = 8


def get_device():
    # Get the appropriate device string for PyTorch operations.
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_audio_data():
    # get all the files, shuffle and return
    sbd = []
    aud = []
    for file in os.listdir(AUDIO_FOLDER):
        if file.endswith('SBD.wav'):
            sbd.append(f'{AUDIO_FOLDER}/{file}')
        else:
            aud.append(f'{AUDIO_FOLDER}/{file}')
    random.shuffle(sbd)
    random.shuffle(aud)
    sbd = sbd[:int(len(sbd) * FILE_USAGE)]
    aud = aud[:int(len(aud) * FILE_USAGE)]
    print(f'* Using {len(sbd)} SBD files, {len(aud)} AUD files')
    return sbd, aud


class AudioIterator:
    def __init__(self, cached=True):
        self.sbd, self.aud = get_audio_data()
        self.index = 0
        self.cached = cached
        if cached is True:
            self.fill_cache()

    def fill_cache(self):
        print('* Caching tensors')
        sbd = []
        for filepath in tqdm(self.sbd):
            sbd.append(self.get_audio(filepath))
        self.sbd = sbd

        aud = []
        for filepath in tqdm(self.aud):
            aud.append(self.get_audio(filepath))
        self.aud = aud

    def __iter__(self):
        return self

    def get_audio(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath)
        # waveform data range is [-1, 1], convert to [0,1]
        waveform += 1.0
        waveform /= 2.0
        return waveform

    def __next__(self):
        max_index = self.index + BATCH_SIZE
        if max_index > len(self.sbd) or max_index > len(self.aud):
            # end of iteration, shuffle faces
            random.shuffle(self.sbd)
            random.shuffle(self.aud)
            self.index = 0
            raise StopIteration

        if self.cached is True:
            sbd_audio = self.sbd[self.index:self.index + BATCH_SIZE]
            aud_audio = self.aud[self.index:self.index + BATCH_SIZE]
        else:
            sbd_audio = [self.get_audio(x) for x in self.sbd[self.index:self.index + BATCH_SIZE]]
            aud_audio = [self.get_audio(x) for x in self.aud[self.index:self.index + BATCH_SIZE]]

        sbd_batch = torch.stack(sbd_audio)
        aud_batch = torch.stack(aud_audio)

        self.index += BATCH_SIZE
        return sbd_batch, aud_batch

    def __len__(self):
        # Get the total number of items in the dataset.
        return len(self.sbd) // BATCH_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # use instancenorm when batch size is 1
        # input -> Conv -> InstanceNorm -> Relu -> Conv -> InstanceNorm
        self.conv = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm1d(channels),
                                  nn.LeakyReLU(),
                                  nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm1d(channels))
        # we don't need a LeakyRELU here as the skip connection already carries over some data
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        # add the original input to the output
        out += residual
        # and a final ReLU after addition - seen to give best results
        return self.relu(out)


class AudioGenerator(nn.Module):
    def __init__(self):
        super(AudioGenerator, self).__init__()
        self.relu = nn.LeakyReLU()

        CHANNEL_MOD = 6

        # Convolutional layers
        # args: channels in, channels out, kernel size etc...
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32 * CHANNEL_MOD, kernel_size=4, stride=4)
        self.ins1 = nn.InstanceNorm1d(32 * CHANNEL_MOD)
        self.conv2 = nn.Conv1d(in_channels=32 * CHANNEL_MOD, out_channels=64 * CHANNEL_MOD, kernel_size=4, stride=4)
        self.ins2 = nn.InstanceNorm1d(64 * CHANNEL_MOD)
        self.conv3 = nn.Conv1d(in_channels=64 * CHANNEL_MOD, out_channels=128 * CHANNEL_MOD, kernel_size=4, stride=4)
        self.ins3 = nn.InstanceNorm1d(128 * CHANNEL_MOD)

        # you can't put resnets in a list because the weight will not be properly
        # transferred to the GPU when moving devices
        self.resnets = nn.Sequential()
        for i in range(TOTAL_RESNETS):
            self.resnets.add_module(f'resnet{i+1}', ResidualBlock(128 * CHANNEL_MOD))

        self.convt1 = nn.ConvTranspose1d(in_channels=128 * CHANNEL_MOD, out_channels=64 * CHANNEL_MOD, kernel_size=4, stride=4)
        self.ins4 = nn.InstanceNorm1d(64 * CHANNEL_MOD)
        self.convt2 = nn.ConvTranspose1d(in_channels=64 * CHANNEL_MOD, out_channels=32 * CHANNEL_MOD, kernel_size=4, stride=4)
        self.ins5 = nn.InstanceNorm1d(32 * CHANNEL_MOD)
        self.convt3 = nn.ConvTranspose1d(in_channels=32 * CHANNEL_MOD, out_channels=1, kernel_size=4, stride=4)

        # Don't we want output values from 0-1? Why are we using tanh?
        # why does that not return less than zero sometimes
        # Because, firstly we use a leaky relu, so almost all info below zero is deleted anyway
        # Sigmoid outputs 0-1, but -1 is not 0; that needs -6 or so
        # tanh outputs -1 to 1, but since we have almost nothing below zero,
        # it's from 0 -> 1, exactly over the range 0 -> 1
        # HOWEVER, note that the warning from matplotlib about clipping input data can happen
        # We only use leaky relu, so some values < 0 could be in the mix
        # Perhaps we should use straight ReLU for the last filter

        # We make the input in the range [-1, 1] but after that a RELU clips the output;
        # the job of the first layer is really to move things into the correct range
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.ins1(self.conv1(x)))
        x = self.relu(self.ins2(self.conv2(x)))
        x = self.relu(self.ins3(self.conv3(x)))

        x = self.resnets(x)

        x = self.relu(self.ins4(self.convt1(x)))
        x = self.relu(self.ins5(self.convt2(x)))
        x = self.relu(self.convt3(x))
        x = self.activate(x)
        return x


class AudioClassifierTestedAudio(nn.Module):
    def __init__(self):
        super(AudioClassifierTestedAudio, self).__init__()
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


def training_loop():
    # https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca

    device = get_device()

    audio_folder = f'{AUDIO_OUTPUT}/{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    disc_sbd = AudioClassifierTestedAudio()
    disc_sbd = disc_sbd.to(device)
    disc_aud = AudioClassifierTestedAudio()
    disc_aud = disc_aud.to(device)

    gen_sbd = AudioGenerator()
    gen_sbd = gen_sbd.to(device)
    gen_aud = AudioGenerator()
    gen_aud = gen_aud.to(device)

    opt_disc = optim.Adam(list(disc_aud.parameters()) + list(disc_sbd.parameters()),
                          lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_aud.parameters()) + list(gen_sbd.parameters()),
                         lr=LEARNING_RATE, betas=(0.5, 0.999))
    #scaler = torch.cuda.amp.GradScaler(enabled=True)

    audio_iterator = AudioIterator()

    # recording loss
    # we want to know how the discriminators are working
    # disc_aud_real / disc_aud_fake in one graph
    # disc_sbd_real / disc_sbd_fake in another graph

    # for the generators, we want to split by generator, so we have
    # loss_gen_sbd, cycle_sbd_loss for 1 cycle
    # loss_gen_aud, cycle_aud_loss for 1 cycle

    # then we need average all of those values

    epoch_losses = [[], [], [], [], [], [], [], []]

    disc_aud.train()
    disc_sbd.train()
    gen_aud.train()
    gen_sbd.train()

    disc_aud_test = []
    disc_sbd_test = []
    gen_loss = []
    cycle_loss = []

    for epoch in range(EPOCHS):
        losses = []

        # grab a sbd and an aud audio
        for sbd_sample, aud_sample in tqdm(audio_iterator):
            sbd_sample = sbd_sample.to(device)
            aud_sample = aud_sample.to(device)

            # make a fake
            fake_aud = gen_aud(sbd_sample)
            # get results of discriminator against the fake
            disc_aud_real = disc_aud(aud_sample)
            # Detach else the weights get mixed up, and we get a double backwards for g_loss later
            # TODO: Understand this a bit better
            # get results of this against the real
            disc_aud_fake = disc_aud(fake_aud.detach())

            # get loss against the real face and the fake one
            # TODO: Here we say an aud is scored 1 and a sbd 0, is that right?
            disc_aud_real_loss = mse(disc_aud_real, torch.ones_like(disc_aud_real))
            disc_aud_fake_loss = mse(disc_aud_fake, torch.zeros_like(disc_aud_fake))
            disc_aud_loss = disc_aud_real_loss + disc_aud_fake_loss

            disc_aud_test.append([disc_aud_real.mean().item(),
                                  disc_aud_fake.mean().item(),
                                  disc_aud_real_loss.mean().item(),
                                  disc_aud_fake_loss.mean().item()])

            # repeat the other way around:
            # start by generating a fake sbd
            fake_sbd = gen_sbd(aud_sample)
            # now get what the discriminator thinks about the real sbd
            disc_sbd_real = disc_sbd(sbd_sample)
            # and also what it thinks about the fake sbd
            disc_sbd_fake = disc_sbd(fake_sbd.detach())
            # now we take the loss both sides: 1 for real and 0 for fake
            disc_sbd_real_loss = mse(disc_sbd_real, torch.ones_like(disc_sbd_real))
            disc_sbd_fake_loss = mse(disc_sbd_fake, torch.zeros_like(disc_sbd_fake))
            # and the loss is now the total of those losses
            disc_sbd_loss = disc_sbd_real_loss + disc_sbd_fake_loss

            # total loss over both discriminators
            # 2.0 is a hyperfactor to slow down the learning
            d_loss = (disc_aud_loss + disc_sbd_loss) / DISCRIMINATOR_SLOWDOWN

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # now train the generators
            # test the discriminators against the generated fakes
            disc_aud_fake = disc_aud(fake_aud)
            disc_sbd_fake = disc_sbd(fake_sbd)
            loss_gen_aud = mse(disc_aud_fake, torch.ones_like(disc_aud_fake))
            loss_gen_sbd = mse(disc_sbd_fake, torch.ones_like(disc_sbd_fake))

            gen_loss.append([loss_gen_aud.mean().item(), loss_gen_sbd.mean().item()])

            # now we convert the fakes back to the correct gender
            # we want this to be the same as the original (that is, seen as the original gender)
            # so let's add that loss as well
            # start by making the images
            cycle_sbd = gen_sbd(fake_aud)
            cycle_aud = gen_aud(fake_sbd)
            cycle_sbd_loss = l1(sbd_sample, cycle_sbd)
            cycle_aud_loss = l1(aud_sample, cycle_aud)

            cycle_loss.append([cycle_sbd_loss.mean().item(), cycle_aud_loss.mean().item()])

            # add up all those losses
            g_loss = loss_gen_sbd + loss_gen_aud + cycle_sbd_loss + cycle_aud_loss

            opt_gen.zero_grad()
            g_loss.backward()
            opt_gen.step()

            # update the losses
            # there are 8 different losses here, from first to last:
            # 1: The loss of the aud discriminator against a real audio
            # 2: The loss of the aud discriminator against a fake audio
            # 3: The loss of the sbd discriminator against a real audio
            # 4: The loss of the sbd discriminator against a fake audio
            # 5: The loss of the sbd -> aud generator
            # 6: The loss of the sbd -> aud -> sbd chain (?)
            # 7: The loss of the aud -> sbd generator
            # 8: The loss of the aud -> sbd -> aud chain (?)
            losses.append([disc_aud_real_loss.mean().item(), disc_aud_fake_loss.mean().item(),
                           disc_sbd_real_loss.mean().item(), disc_sbd_fake_loss.mean().item(),
                           loss_gen_aud.mean().item(), cycle_aud_loss.mean().item(),
                           loss_gen_sbd.mean().item(), cycle_sbd_loss.mean().item()])

        # update all losses
        averages = []
        size = len(audio_iterator)
        # 8 pieces of data in all
        for i in range(8):
            average = sum([x[i] for x in losses]) / size
            averages.append(average)
            epoch_losses[i].append(average)

        print(f'\nE #{str(epoch+1).zfill(3)}: DF: {averages[0]:.2f}, {averages[1]:.2f}, DM: {averages[2]:.2f}, DF: {averages[3]:.2f}')
        print(f'        GF: {averages[4]:.2f}, {averages[5]:.2f}, GM: {averages[6]:.2f}, DF: {averages[7]:.2f}')

        # each of these is of dimension (BATCH_SIZE, 3, 128, 128); we just need the first one
        # we also need the "real" image, to compare to
        audio = [aud_sample[0].detach().to('cpu'),
                 fake_sbd[0].detach().to('cpu')]
        save_audio(audio, audio_folder, epoch)

    test_plot(disc_aud_test)
    #show_gen_loss(gen_loss)
    #show_cycle_loss(cycle_loss)
    #plot_graphs(epoch_losses)


def save_audio(tensors, folder_name, epoch):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    aud_path = f'{folder_name}/epoch_{epoch}_aud.wav'
    sbd_path = f'{folder_name}/epoch_{epoch}_sbd.wav'

    # the tensors have values between 0 -> 1, convert this to -1 -> +1
    tensors[0] *= 2.0
    tensors[0] -= 1.0
    tensors[1] *= 2.0
    tensors[1] -= 1.0

    torchaudio.save(aud_path, tensors[0], 44100)
    torchaudio.save(sbd_path, tensors[1], 44100)


def plot_graphs(data):
    print(len(data))
    labels = ['AUD Disc: Real', 'AUD Disc: Fake', 'SBD Disc: Real', 'SBD Disc: Fake', 'AUD Gen', 'SBD Gen', 'AUD Cycle', 'SBD Cycle']
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()

    # Plot first two lists on the first graph
    for i in range(2):
        axs[0].plot(data[i], label=labels[i])
    axs[0].legend()

    # Plot next two lists on the second graph
    for i in range(2, 4):
        axs[1].plot(data[i], label=labels[i])
    axs[1].legend()

    # Plot the last four lists on individual graphs
    for i in range(4, 8):
        axs[i - 2].plot(data[i - 2], label=labels[i])
        axs[i - 2].legend()

    # Improve layout
    plt.tight_layout()
    datestamp = datetime.now().strftime("%Y_%b_%d_%H_%M")
    # Example: Create a filename for a .png file
    filepath = f'{IMAGE_OUTPUT}/{datestamp}_audio_E{EPOCHS}_FU{int(FILE_USAGE * 100)}.png'
    plt.savefig(filepath)
    # Show the plots
    plt.show()


def test_plot(metrics):
    real_values = [x[0] for x in metrics]
    fake_values = [x[1] for x in metrics]
    real_loss = [x[2] for x in metrics]
    fake_loss = [x[3] for x in metrics]
    # Create the plot
    plt.plot(real_values, label='Real Values', color='red')
    plt.plot(fake_values, label='Fake Values', color='blue')
    plt.plot(real_loss, label='Real Loss', color='orange')
    plt.plot(fake_loss, label='Fake Loss', color='purple')

    plt.legend()
    # Show the plot
    plt.show()


def show_gen_loss(metrics):
    aud_loss = [x[0] for x in metrics]
    sbd_loss = [x[1] for x in metrics]
    plt.plot(aud_loss, label='Aud Loss', color='red')
    plt.plot(sbd_loss, label='SBD Loss', color='blue')
    plt.legend()
    plt.show()


def show_cycle_loss(metrics):
    sbd_loss = [x[0] for x in metrics]
    aud_loss = [x[1] for x in metrics]
    plt.plot(sbd_loss, label='SBD Cycle', color='red')
    plt.plot(aud_loss, label='AUD Cycle', color='blue')
    plt.legend()
    plt.show()


def plot_graphs_old(metrics):
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
        datestamp = datetime.now().strftime("%Y_%b_%d_%H_%M")
        # Example: Create a filename for a .png file
        filepath = f'{IMAGE_OUTPUT}/{datestamp}_audio_E{EPOCHS}_FU{int(FILE_USAGE * 100)}.png'
        plt.savefig(filepath)
        plt.show()


if __name__ == '__main__':
    #model = AudioGenerator().cuda()
    #summary(model, input_size=(1, 65536), batch_size=-1)
    training_loop()

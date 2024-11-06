import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import optim
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity


# This is based on the original male <-> female Cyclegan, just adjusted for the
# cyclegan tensors and renamed some variables.
# Largest difference might be the input tensor which is [512, 512, 1]


SPECTROGRAM_FOLDER = Path('/home/sparky/data/MLData/GD_MEL')
IMAGE_OUTPUT = '/home/sparky/data/code/dead-cyclegan/output/spectrograms'
MODEL_DIRECTORY = Path.cwd() / 'models'

IMAGE_SIZE = 512
TEST_RATIO = 0.2
FILE_USAGE = 0.05

EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
TOTAL_RESNETS = 5


def get_device():
    # Get the appropriate device string for PyTorch operations.
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_files():
    all_dirs = os.listdir(SPECTROGRAM_FOLDER)
    aud_files = []
    sbd_files = []
    for i in all_dirs:
        show_dir = SPECTROGRAM_FOLDER / i
        show_files = [show_dir / x for x in os.listdir(show_dir)]
        if str(i).endswith('SBD'):
            sbd_files.extend(show_files)
        else:
            aud_files.extend(show_files)
    # make them the same length
    shorter_length = min(int(len(sbd_files) * FILE_USAGE), int(len(aud_files) * FILE_USAGE))
    # shuffle the files to prevent the same ones all the time
    random.shuffle(sbd_files)
    random.shuffle(aud_files)
    sbd_files = sbd_files[:shorter_length]
    aud_files = aud_files[:shorter_length]
    print(f'* Using {shorter_length * 2} total files')
    return sbd_files, aud_files


class ImageIterator:
    def __init__(self, cached=True):
        sbd_files, aud_files = get_files()
        self.sbd_files = sbd_files
        self.aud_files = aud_files
        self.index = 0
        self.cached = cached
        if cached is True:
            self.fill_cache()

    def fill_cache(self):
        print('* Caching tensors')
        audio_files = []
        for i in tqdm(self.sbd_files):
            audio_files.append(self.load_tensor(i))
        self.sbd_files = audio_files

        audio_files = []
        for i in tqdm(self.aud_files):
            audio_files.append(self.load_tensor(i))
        self.aud_files = audio_files

    def load_tensor(self, filename):
        data = np.float32(np.load(filename))
        # turn [512, 512] tensor to [1, 512, 512]
        return torch.from_numpy(data).unsqueeze(0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index + BATCH_SIZE > len(self.sbd_files):
            # end of iteration, shuffle faces
            random.shuffle(self.sbd_files)
            random.shuffle(self.aud_files)
            self.index = 0
            raise StopIteration

        if self.cached is True:
            sbd_tensors = self.sbd_files[self.index:self.index + BATCH_SIZE]
            aud_tensors = self.aud_files[self.index:self.index + BATCH_SIZE]
        else:
            sbd_tensors = [self.load_tensor(x) for x in self.sbd_files[self.index:self.index + BATCH_SIZE]]
            aud_tensors = [self.load_tensor(x) for x in self.aud_files[self.index:self.index + BATCH_SIZE]]

        sbd_batch = torch.stack(sbd_tensors)
        aud_batch = torch.stack(aud_tensors)

        self.index += BATCH_SIZE
        return sbd_batch, aud_batch

    def __len__(self):
        # Get the total number of items in the dataset when batched
        return len(self.sbd_files) // BATCH_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # Batch size is 1, so use instancenorm
        # input -> Conv -> InstanceNorm -> Relu -> Conv -> InstanceNorm
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(channels),
                                  nn.ReLU(),
                                  nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        # add the original input to the output
        out += residual
        # and a final ReLU after addition - seen to give best results
        return self.relu(out)


class ImageGenerator(nn.Module):
    def __init__(self):
        super(ImageGenerator, self).__init__()

        self.relu = nn.LeakyReLU()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.ins1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.ins2 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.ins3 = nn.InstanceNorm2d(128)

        # you can't put resnets in a list because the weight will not be properly
        # transferred to the GPU when moving devices
        self.resnets = nn.Sequential()
        for i in range(TOTAL_RESNETS):
            self.resnets.add_module(f'resnet{i+1}', ResidualBlock(128))

        self.convt1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.ins4 = nn.InstanceNorm2d(64)
        self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.ins5 = nn.InstanceNorm2d(32)
        self.convt3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Don't we want output values from 0-1? Why are we using tanh?
        # why does that not return less than zero sometimes
        # Because, firstly we use a leaky relu, so almost all info below zero is deleted anyway
        # Sigmoid outputs 0-1, but -1 is not 0; that needs -6 or so
        # tanh outputs -1 to 1, but since we have almost nothing below zero,
        # it's from 0 -> 1, exactly over the range 0 -> 1
        # HOWEVER, note that the warning from matplotlib about clipping input data can happen
        # We only use leaky relu, so some values < 0 could be in the mix
        # Perhaps we should use straight ReLU for the last filter
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


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        # The size of the input features to the first fully connected layer is 16*16*128.
        # This is because the original image size is 512x512, and it goes through three
        # convolution layers and three pooling layers, each reducing the size by half.
        self.fc1 = nn.Linear(524288, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 524288)

        # Apply fully connected layers with ReLU
        x = F.relu(self.fc1(x))

        # Output layer with a single neuron for binary classification
        x = torch.sigmoid(self.fc2(x))

        return x


def training_loop():
    # https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca

    device = get_device()

    image_folder = f'{IMAGE_OUTPUT}/{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    disc_sbd = ImageClassifier()
    disc_sbd = disc_sbd.to(device)
    disc_aud = ImageClassifier().to(device)
    disc_aud = disc_aud.to(device)

    gen_sbd = ImageGenerator()
    gen_sbd = gen_sbd.to(device)
    gen_aud = ImageGenerator()
    gen_aud = gen_aud.to(device)

    opt_disc = optim.Adam(list(disc_aud.parameters()) + list(disc_sbd.parameters()),
                          lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_aud.parameters()) + list(gen_sbd.parameters()),
                         lr=LEARNING_RATE, betas=(0.5, 0.999))
    #scaler = torch.cuda.amp.GradScaler(enabled=True)

    tensor_iterator = ImageIterator()

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

    for epoch in range(EPOCHS):
        losses = []

        # grab a sbd and an aud tensor
        for sbd_tensor, aud_tensor in tqdm(tensor_iterator):
            sbd_tensor = sbd_tensor.to(device)
            aud_tensor = aud_tensor.to(device)

            # make a fake
            fake_aud = gen_aud(sbd_tensor)
            disc_aud_real = disc_aud(aud_tensor)
            # Detach else the weights get mixed up, and we get a double backwards for g_loss later
            # TODO: Understand this a bit better
            disc_aud_fake = disc_aud(fake_aud.detach())

            # get loss against the real face and the fake one
            disc_aud_real_loss = mse(disc_aud_real, torch.ones_like(disc_aud_real))
            disc_aud_fake_loss = mse(disc_aud_fake, torch.zeros_like(disc_aud_fake))
            disc_aud_loss = disc_aud_real_loss + disc_aud_fake_loss

            # repeat the other way around
            fake_sbd = gen_sbd(aud_tensor)
            disc_sbd_real = disc_sbd(sbd_tensor)
            disc_sbd_fake = disc_sbd(fake_sbd.detach())
            disc_sbd_real_loss = mse(disc_sbd_real, torch.ones_like(disc_sbd_real))
            disc_sbd_fake_loss = mse(disc_sbd_fake, torch.zeros_like(disc_sbd_fake))
            disc_sbd_loss = disc_sbd_real_loss + disc_sbd_fake_loss

            # total loss over both discriminators
            # 2.0 is a hyperfactor to slow down the learning
            d_loss = (disc_aud_loss + disc_sbd_loss) / 2.0

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # now train the generators
            # test the discriminators against the generated fakes
            disc_aud_fake = disc_aud(fake_aud)
            disc_sbd_fake = disc_sbd(fake_sbd)
            loss_gen_aud = mse(disc_aud_fake, torch.ones_like(disc_aud_fake))
            loss_gen_sbd = mse(disc_sbd_fake, torch.ones_like(disc_sbd_fake))

            # now we convert the fakes back to the correct gender
            # we want this to be the same as the original (that is, seen as the original gender)
            # so let's add that loss as well
            # start by making the images
            cycle_sbd = gen_sbd(fake_aud)
            cycle_aud = gen_aud(fake_sbd)
            cycle_sbd_loss = l1(sbd_tensor, cycle_sbd)
            cycle_aud_loss = l1(aud_tensor, cycle_aud)

            # add up all those losses
            g_loss = loss_gen_sbd + loss_gen_aud + cycle_sbd_loss + cycle_aud_loss

            opt_gen.zero_grad()
            g_loss.backward()
            opt_gen.step()

            # update the losses
            losses.append([disc_aud_real_loss.mean().item(), disc_aud_fake_loss.mean().item(),
                           disc_sbd_real_loss.mean().item(), disc_sbd_fake_loss.mean().item(),
                           loss_gen_aud.mean().item(), cycle_aud_loss.mean().item(),
                           loss_gen_sbd.mean().item(), cycle_sbd_loss.mean().item()])

        # update all losses
        averages = []
        size = len(tensor_iterator)
        # 8 pieces of data in all
        # 8 pieces of data in all
        for i in range(8):
            average = sum([x[i] for x in losses]) / size
            averages.append(average)
            epoch_losses[i].append(average)

        print(f'E #{str(epoch+1).zfill(3)}: DF: {averages[0]:.2f}, {averages[1]:.2f}, DM: {averages[2]:.2f}, DF: {averages[3]:.2f}')
        print(f'        GF: {averages[4]:.2f}, {averages[5]:.2f}, GM: {averages[6]:.2f}, DF: {averages[7]:.2f}')

        # each of these is of dimension (BATCH_SIZE, 1, 128, 128); we just need the first one
        # we also need the "real" image, to compare to
        tensors = [sbd_tensor[0].detach().to('cpu'),
                   fake_aud[0].detach().to('cpu'),
                   cycle_sbd[0].detach().to('cpu'),
                   aud_tensor[0].detach().to('cpu'),
                   fake_sbd[0].detach().to('cpu'),
                   cycle_aud[0].detach().to('cpu')]
        save_tensors(tensors, image_folder)

    save_model(gen_sbd)
    plot_graphs(epoch_losses)


def save_tensors(tensors, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    index = len(os.listdir(folder_name)) + 1
    fullpath = f'{folder_name}/epoch_{index}.png'

    labels = ['Real SBD', 'Fake AUD', 'Cycle SBD', 'Real AUD', 'Fake SBD', 'Cycle AUD']
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')

    for i, tensor in enumerate(tensors):
        # Convert tensor to numpy and reshape to H x W x C format
        tensor = tensor.permute(1, 2, 0).numpy()
        axs[i].imshow(tensor)
        axs[i].set_title(labels[i])
    print(fullpath)
    plt.savefig(fullpath)


def save_model(aud_to_sbd_model):
    # filename is YY_MM_DD_HH.pth
    now = datetime.now()
    filename = f'{str(now.year)[:2]}_{now.month:02d}_{now.day:02d}_{now.hour:02d}.pth'
    filepath = MODEL_DIRECTORY / filename
    torch.save(aud_to_sbd_model.state_dict(), filepath)
    print(f'* Saved model to {filepath}')


def plot_graphs(data):
    print(len(data))
    labels = ['F Disc: Real', 'F Disc: Fake', 'M Disc: Real', 'M Disc: Fake', 'F Gen', 'M Gen', 'F Cycle', 'M Cycle']
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


def test_model_input():
    model = ImageGenerator()
    data = ImageIterator()
    male, female = next(iter(data))
    print(male.shape)
    render1 = model(male)
    print(render1.shape)


def test_display():
    tensors = [torch.rand(1, 512, 512) for x in range(6)]
    save_tensors(tensors, IMAGE_OUTPUT)


if __name__ == '__main__':
    #test_model_input()
    #model = ImageGenerator().cuda()
    #model = ImageClassifier().cuda()
    #summary(model, input_size=(1, 512, 512), batch_size=-1)
    #test_display()
    training_loop()

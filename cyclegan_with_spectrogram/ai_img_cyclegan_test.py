import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch import optim
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity


# This is a sample image discriminator
# We will use almost the same code in the audio discriminator
# this code is used as a test to ensure everything is correct


IMAGE_FOLDER = '/home/sparky/data/MLData/CelebA/images'
ATTR_FILE = '/home/sparky/data/MLData/CelebA/attributes.txt'
IMAGE_OUTPUT = '/home/sparky/data/code/dead-cyclegan/output/image_results'

MALE_INDEX = 21
IMAGE_SIZE = 128
TEST_RATIO = 0.2
FILE_USAGE = 0.1

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
TOTAL_RESNETS = 5


def get_device():
    # Get the appropriate device string for PyTorch operations.
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_image_data():
    all_files = os.listdir(IMAGE_FOLDER)
    random.shuffle(all_files)
    all_files = all_files[:int(len(all_files) * FILE_USAGE)]
    print(f'* Using {len(all_files)} files')
    return all_files


def get_faces(files):
    attributes = get_attributes()
    males = []
    females = []
    for i in files:
        value = i.split('.')[0]
        if value not in attributes:
            pass
        else:
            if attributes[value] > 0:
                males.append(i)
            else:
                females.append(i)

    # make them the same length
    shorter_length = min(len(males), len(females))
    males = males[:shorter_length]
    females = females[:shorter_length]
    return males, females


def get_attributes():
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


class ImageIterator:
    def __init__(self, cached=True):
        files = get_image_data()
        male_faces, female_faces = get_faces(files)
        self.male_faces = male_faces
        self.female_faces = female_faces
        self.index = 0
        self.cached = cached
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()])
        if cached is True:
            self.fill_cache()

    def fill_cache(self):
        print('* Caching tensors')
        males = []
        for i in tqdm(self.male_faces):
            males.append(self.get_image(i))
        self.male_faces = males

        females = []
        for i in tqdm(self.female_faces):
            females.append(self.get_image(i))
        self.female_faces = females

    def __iter__(self):
        return self

    def get_image(self, filename):
        image = Image.open(f'{IMAGE_FOLDER}/{filename}')
        return self.transform(image)

    def __next__(self):
        if self.index + BATCH_SIZE > len(self.male_faces):
            # end of iteration, shuffle faces
            random.shuffle(self.male_faces)
            random.shuffle(self.female_faces)
            self.index = 0
            raise StopIteration

        if self.cached is True:
            male_faces = self.male_faces[self.index:self.index + BATCH_SIZE]
            female_faces = self.female_faces[self.index:self.index + BATCH_SIZE]
        else:
            male_faces = [self.get_image(x) for x in self.male_faces[self.index:self.index + BATCH_SIZE]]
            female_faces = [self.get_image(x) for x in self.female_faces[self.index:self.index + BATCH_SIZE]]

        male_batch = torch.stack(male_faces)
        female_batch = torch.stack(female_faces)

        self.index += BATCH_SIZE
        return male_batch, female_batch

    def __len__(self):
        # Get the total number of items in the dataset.
        return len(self.male_faces) // BATCH_SIZE


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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.ins1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.ins2 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.ins3 = nn.InstanceNorm2d(128)

        # you can't put resnets in a list because the weight will not be properly
        # transferred to the GPU when moving devices
        self.resnets = nn.Sequential()
        for i in range(TOTAL_RESNETS):
            self.resnets.add_module(f'resnet{i+1}', ResidualBlock(128))

        self.convt1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.ins4 = nn.InstanceNorm2d(64)
        self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.ins5 = nn.InstanceNorm2d(32)
        self.convt3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

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


def training_loop():
    # https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca

    device = get_device()

    image_folder = f'{IMAGE_OUTPUT}/{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    disc_male = ImageClassifier()
    disc_male = disc_male.to(device)
    disc_female = ImageClassifier().to(device)
    disc_female = disc_female.to(device)

    gen_male = ImageGenerator()
    gen_male = gen_male.to(device)
    gen_female = ImageGenerator()
    gen_female = gen_female.to(device)

    opt_disc = optim.Adam(list(disc_female.parameters()) + list(disc_male.parameters()),
                          lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_female.parameters()) + list(gen_male.parameters()),
                         lr=LEARNING_RATE, betas=(0.5, 0.999))
    #scaler = torch.cuda.amp.GradScaler(enabled=True)

    faces_iterator = ImageIterator()

    # recording loss
    # we want to know how the discriminators are working
    # disc_female_real / disc_female_fake in one graph
    # disc_male_real / disc_male_fake in another graph

    # for the generators, we want to split by generator, so we have
    # loss_gen_male, cycle_male_loss for 1 cycle
    # loss_gen_female, cycle_female_loss for 1 cycle

    # then we need average all of those values

    epoch_losses = [[], [], [], [], [], [], [], []]

    disc_female.train()
    disc_male.train()
    gen_female.train()
    gen_male.train()

    for epoch in range(EPOCHS):
        losses = []

        # grab a male and a female face
        for male_face, female_face in tqdm(faces_iterator):
            male_face = male_face.to(device)
            female_face = female_face.to(device)

            # make a fake
            fake_female = gen_female(male_face)
            disc_female_real = disc_female(female_face)
            # Detach else the weights get mixed up, and we get a double backwards for g_loss later
            # TODO: Understand this a bit better
            disc_female_fake = disc_female(fake_female.detach())

            # get loss against the real face and the fake one
            disc_female_real_loss = mse(disc_female_real, torch.ones_like(disc_female_real))
            disc_female_fake_loss = mse(disc_female_fake, torch.zeros_like(disc_female_fake))
            disc_female_loss = disc_female_real_loss + disc_female_fake_loss

            # repeat the other way around
            fake_male = gen_male(female_face)
            disc_male_real = disc_male(male_face)
            disc_male_fake = disc_male(fake_male.detach())
            disc_male_real_loss = mse(disc_male_real, torch.ones_like(disc_male_real))
            disc_male_fake_loss = mse(disc_male_fake, torch.zeros_like(disc_male_fake))
            disc_male_loss = disc_male_real_loss + disc_male_fake_loss

            # total loss over both discriminators
            # 2.0 is a hyperfactor to slow down the learning
            d_loss = (disc_female_loss + disc_male_loss) / 2.0

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # now train the generators
            # test the discriminators against the generated fakes
            disc_female_fake = disc_female(fake_female)
            disc_male_fake = disc_male(fake_male)
            loss_gen_female = mse(disc_female_fake, torch.ones_like(disc_female_fake))
            loss_gen_male = mse(disc_male_fake, torch.ones_like(disc_male_fake))

            # now we convert the fakes back to the correct gender
            # we want this to be the same as the original (that is, seen as the original gender)
            # so let's add that loss as well
            # start by making the images
            cycle_male = gen_male(fake_female)
            cycle_female = gen_female(fake_male)
            cycle_male_loss = l1(male_face, cycle_male)
            cycle_female_loss = l1(female_face, cycle_female)

            # add up all those losses
            g_loss = loss_gen_male + loss_gen_female + cycle_male_loss + cycle_female_loss

            opt_gen.zero_grad()
            g_loss.backward()
            opt_gen.step()

            # update the losses
            losses.append([disc_female_real_loss.mean().item(), disc_female_fake_loss.mean().item(),
                           disc_male_real_loss.mean().item(), disc_male_fake_loss.mean().item(),
                           loss_gen_female.mean().item(), cycle_female_loss.mean().item(),
                           loss_gen_male.mean().item(), cycle_male_loss.mean().item()])

        # update all losses
        averages = []
        size = len(faces_iterator)
        # 8 pieces of data in all
        # 8 pieces of data in all
        for i in range(8):
            average = sum([x[i] for x in losses]) / size
            averages.append(average)
            epoch_losses[i].append(average)

        print(f'E #{str(epoch+1).zfill(3)}: DF: {averages[0]:.2f}, {averages[1]:.2f}, DM: {averages[2]:.2f}, DF: {averages[3]:.2f}')
        print(f'        GF: {averages[4]:.2f}, {averages[5]:.2f}, GM: {averages[6]:.2f}, DF: {averages[7]:.2f}')

        # each of these is of dimension (BATCH_SIZE, 3, 128, 128); we just need the first one
        # we also need the "real" image, to compare to
        images = [male_face[0].detach().to('cpu'),
                  fake_female[0].detach().to('cpu'),
                  cycle_male[0].detach().to('cpu'),
                  female_face[0].detach().to('cpu'),
                  fake_male[0].detach().to('cpu'),
                  cycle_female[0].detach().to('cpu')]
        save_images(images, image_folder)

    plot_graphs(epoch_losses)


def save_images(tensors, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    index = len(os.listdir(folder_name)) + 1
    fullpath = f'{folder_name}/epoch_{index}.png'

    labels = ['Real Male', 'Fake Female', 'Cycle Male', 'Real Female', 'Fake Male', 'Cycle Female']
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')

    for i, tensor in enumerate(tensors):
        # Convert tensor to numpy and reshape to H x W x C format
        tensor = tensor.permute(1, 2, 0).numpy()
        axs[i].imshow(tensor)
        axs[i].set_title(labels[i])
    plt.savefig(fullpath)


def display_tensors(tensors):
    # display tensors
    labels = ['Fake Male', 'Fake Female', 'Cycle Male', 'Cycle Female']
    fig, axs = plt.subplots(2, 2, figsize=(5 * 2, 5 * 2))

    # Flatten the axs array for easy iteration and remove axes
    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')

    # Display each tensor
    for i, tensor in enumerate(tensors):
        # Convert tensor to numpy and reshape to H x W x C format
        tensor = tensor.permute(1, 2, 0).numpy()
        axs[i].imshow(tensor)
        axs[i].set_title(labels[i])

    plt.show()


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


def plot_graphs_old(metrics):
    labels = ['Disc Female', 'Disc Male', 'Gen Female', 'Gen Male']
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        row = i // 2
        col = i % 2
        x_values = [x for x in range(len(metrics))]
        y_values1 = [x[i * 2] for x in metrics]  # Values for the first index in the pair
        y_values2 = [x[i * 2 + 1] for x in metrics]  # Values for the second index in the pair
        axs[row, col].plot(x_values, y_values1)
        axs[row, col].plot(x_values, y_values2)
        axs[row, col].set_title(labels[0])
    plt.tight_layout()
    plt.show()


def test_display():
    tensors = [torch.rand(3, 128, 128),
               torch.rand(3, 128, 128),
               torch.rand(3, 128, 128),
               torch.rand(3, 128, 128)]
    display_tensors(tensors)


def test_plot():
    data = [[1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10]]
    plot_graphs(data)


def test_model_input():
    model = ImageGenerator()
    data = ImageIterator()
    male, female = next(iter(data))
    print(male.shape)
    render1 = model(male)
    print(render1.shape)


def test_ram_usage():
    disc_a = ImageClassifier()
    gen_a = ImageGenerator()
    inputs = torch.rand(16, 3, 128, 128)

    with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
        disc_a(inputs)
    print(prof.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=10))

    with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
        gen_a(inputs)
    print(prof.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=10))


if __name__ == '__main__':
    #test_display()
    #test_plot()
    #test_model_input()
    #test_ram_usage()

    #model = ImageGenerator().cuda()
    #summary(model, input_size=(3, 256, 256), batch_size=-1)
    #summary(model, input_size=(3, 256, 256), batch_size=-1)
    training_loop()

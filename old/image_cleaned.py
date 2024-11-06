import numpy as np

from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import os
import glob
import itertools
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# This is the code from
# https://www.kaggle.com/code/songseungwon/cyclegan-tutorial-from-scratch-monet-to-photo
# but tided up and improved. Goals:

# Improve quality
# Ensure files are from correct place
# Save image results as files
# Save model as separate file
# Show graphs after run

TOTAL_CPUS = 2
ROOT_FOLDER = './data/monet'
OUTPUT_FOLDER = Path('./output/monet')
MODEL_FOLDER = Path('./output/monet/models')

# image characteristics
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3

# epoch to start, total epochs
EPOCH = 0
TOTAL_EPOCHS = 10
BATCH_SIZE = 2
# adam learning rate, TODO: decay of first order momentum of gradient
LEARNING_RATE = 0.0002         # adam : learning rate
B1 = 0.5
B2 = 0.999
# TODO: suggested default : 100 (suggested 'n_epochs' is 200) - epoch from which to start lr decay
DECAY_EPOCH = 4
# what amount of files to test
TEST_RATIO = 0.2


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # Pads the input tensor using the reflection of the input boundary
            # TODO: Presume because of conv2d?
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            # TODO: LeakyRELU better?
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features))

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial Convolution Block
        out_features = 64
        model = [nn.ReflectionPad2d(channels),
                 # TODO: Why 7?
                 nn.Conv2d(channels, out_features, 7),
                 nn.InstanceNorm2d(out_features),
                 nn.ReLU(inplace=True)]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [nn.Upsample(scale_factor=2),  # --> width * 2, height * 2
                      nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      nn.ReLU(inplace=True)]
            in_features = out_features

        # Output Layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()]

        # Unpacking
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            # Returns down- sampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                # TODO: Why InstanceNorm and not BatchNorm?
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # * unpacks the sequence returned by the function
            # so bar(*range(3)) would be bar(0, 1, 2) and NOT bar([0, 1, 2])
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1))

    def forward(self, img):
        return self.model(img)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # reset Conv2d's weight(tensor) with Gaussian Distribution
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            # reset Conv2d's bias(tensor) with Constant(0)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # reset BatchNorm2d's weight(tensor) with Gaussian Distribution
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            # reset BatchNorm2d's bias(tensor) with Constant(0)
            torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    # learning rate scheduler setting
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def to_rgb(image):
    # convert image to RGB if it is not already
    rgb_image = Image.new('RGB', image.size)
    rgb_image.paste(image)
    return rgb_image


def get_timestamped_foldername():
    time_now = datetime.now()
    return f"{time_now.strftime('%-d-%m-%y_%H:%M:%S')}"


def get_image_folder():
    # create an image folder timestamped to when we started
    folder_name = get_timestamped_foldername()
    # create folder or raise error if folder exists
    folder_path = OUTPUT_FOLDER / 'images' / folder_name
    if os.path.exists(folder_path):
        raise AssertionError(f'Folder {str(folder_path)} already exists - aborting')
    # create the folder and return the path
    os.mkdir(folder_path)
    return folder_path


def sample_images(image_folder, current_epoch, G_AB, G_BA, data):
    # Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # save a generated sample from the test set for this epoch number
    imgs = next(iter(data))
    # switch off learning
    G_AB.eval()
    G_BA.eval()
    # A : monet
    real_a = imgs['A'].type(torch.cuda.FloatTensor)
    fake_b = G_AB(real_a).detach()
    # B : photo
    real_b = imgs['B'].type(torch.cuda.FloatTensor)
    fake_a = G_BA(real_b).detach()
    # Arange images along x-axis
    real_a = make_grid(real_a, nrow=5, normalize=True)
    fake_b = make_grid(fake_b, nrow=5, normalize=True)
    real_b = make_grid(real_b, nrow=5, normalize=True)
    fake_a = make_grid(fake_a, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_a, fake_b, real_b, fake_a), 1)
    plt.imshow(image_grid.cpu().permute(1,2,0))
    plt.title('Real A vs Fake B | Real B vs Fake A')
    plt.axis('off')
    plt.savefig(image_folder / f'epoch_{current_epoch}.png')


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        set_a = sorted(glob.glob(os.path.join(root + f'/{mode}A') + '/*.*'))
        set_b = sorted(glob.glob(os.path.join(root + f'/{mode}B') + '/*.*'))
        self.files_A = set_a[:int(len(set_a) * TEST_RATIO)]
        self.files_B = set_b[:int(len(set_b) * TEST_RATIO)]

    def __getitem__(self, index):
        image_a = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_b = Image.open(self.files_B[np.random.randint(0, len(self.files_B) - 1)])
        else:
            image_b = Image.open(self.files_B[index % len(self.files_B)])
        if image_a.mode != 'RGB':
            image_a = to_rgb(image_a)
        if image_b.mode != 'RGB':
            image_b = to_rgb(image_b)

        item_a = self.transform(image_a)
        item_b = self.transform(image_b)
        return {'A': item_a, 'B': item_b}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class TotalLoss:
    def __init__(self):
        self.d_loss = []
        self.g_loss = []
        self.adverserial = []
        self.cycle = []
        self.identity = []

    def add_loss(self, d, g, a, c, i):
        self.d_loss.append(d)
        self.g_loss.append(g)
        self.adverserial.append(a)
        self.cycle.append(c)
        self.identity.append(i)

    def render_and_save(self, image_folder):
        px = 1 / plt.rcParams['figure.dpi']
        fig = plt.figure(layout='constrained', figsize=(1200 * px, 800 * px))
        gs = GridSpec(ncols=6, nrows=2, figure=fig)
        fig.suptitle('Results', fontsize=14)

        disc = fig.add_subplot(gs[0, :3])
        gens = fig.add_subplot(gs[0, 3:])
        advs = fig.add_subplot(gs[1, :2])
        cycl = fig.add_subplot(gs[1, 2:4])
        idnt = fig.add_subplot(gs[1, 4:])

        disc.plot([x[0] for x in self.d_loss], label='Disc A loss')
        disc.plot([x[1] for x in self.d_loss], label='Disc B loss')
        disc.plot([x[2] for x in self.d_loss], label='Disc Total loss')
        disc.set_title('Discriminators')

        gens.plot([x[0] for x in self.g_loss], label='Gen A -> B loss')
        gens.plot([x[1] for x in self.g_loss], label='Gen B -> A loss')
        gens.plot([x[2] for x in self.g_loss], label='Gen Total loss')
        gens.set_title('Generators')

        advs.plot(self.adverserial, label='Adversarial loss')
        advs.set_title('Adverserial')
        cycl.plot(self.cycle, label='Cycle loss')
        cycl.set_title('Cycle')
        idnt.plot(self.identity, label='Identity loss')
        idnt.set_title('Identity')

        plt.savefig(image_folder / 'results.png')
        plt.show()


def save_model(img_to_monet_model, filename):
    # filename is YY_MM_DD_HH.pth
    filepath = MODEL_FOLDER / filename
    torch.save(img_to_monet_model.state_dict(), filepath)
    print(f'* Saved model to {filepath}')


def image_cyclegan():
    # define loss functions
    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_cycle = torch.nn.L1Loss().cuda()
    criterion_identity = torch.nn.L1Loss().cuda()

    # create the generator and discriminator
    input_shape = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    # suggested default, number of residual blocks in generator
    n_residual_blocks = 9

    # generator A to B, where A is monet and B is a photo
    G_AB = GeneratorResNet(input_shape, n_residual_blocks).cuda()
    G_BA = GeneratorResNet(input_shape, n_residual_blocks).cuda()
    D_A = Discriminator(input_shape).cuda()
    D_B = Discriminator(input_shape).cuda()

    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # configure optimisers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LEARNING_RATE, betas=(B1, B2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LEARNING_RATE, betas=(B1, B2))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(TOTAL_EPOCHS, EPOCH, DECAY_EPOCH).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(TOTAL_EPOCHS, EPOCH, DECAY_EPOCH).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(TOTAL_EPOCHS, EPOCH, DECAY_EPOCH).step)

    transforms_ = [transforms.Resize(int(IMAGE_HEIGHT * 1.12), Image.BICUBIC),
                   transforms.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


    dataloader = DataLoader(ImageDataset(ROOT_FOLDER, transforms_=transforms_, unaligned=True),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=TOTAL_CPUS)

    val_dataloader = DataLoader(ImageDataset(ROOT_FOLDER, transforms_=transforms_, unaligned=True, mode='test'),
                                batch_size=5,
                                shuffle=True,
                                num_workers=TOTAL_CPUS)

    image_folder = get_image_folder()
    losses = TotalLoss()

    for epoch in range(EPOCH, TOTAL_EPOCHS):
        # if cuda else torch.Tensor, but we are in cuda land

        for i, batch in enumerate(tqdm(dataloader)):

            # Set model input
            real_A = batch['A'].type(torch.cuda.FloatTensor)
            real_B = batch['B'].type(torch.cuda.FloatTensor)

            # Adversarial ground truths
            # both of these need require_grad=False, which is the default in 2.0
            valid = torch.cuda.FloatTensor(np.ones((real_A.size(0), *D_A.output_shape)))
            fake = torch.cuda.FloatTensor(np.zeros((real_A.size(0), *D_A.output_shape)))

            # put generators in training mode
            G_AB.train()
            G_BA.train()

            # TODO: Integrated optimizer(G_AB, G_BA)
            optimizer_G.zero_grad()

            # Identity Loss
            # If you put A into a generator that creates A with B,
            # then of course A must come out as it is.
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            # Taking this into consideration, add an identity loss that simply compares 'A and A' (or 'B and B').
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN Loss
            # fake_B is fake-photo that generated by real monet-drawing
            fake_B = G_AB(real_A)
            # tricking the 'fake-B' into 'real-B'
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            # tricking the 'fake-A' into 'real-A'
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle Loss
            # recov_A is fake-monet-drawing that generated by fake-photo
            recov_A = G_BA(fake_B)
            # Reduces the difference between the restored image and the real image
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # ------> Total Loss
            # multiply suggested weight(default cycle loss weight : 10, default identity loss weight : 5)
            # TODO: Increases the gradient?
            loss_G = loss_GAN + (10.0 * loss_cycle) + (5.0 * loss_identity)

            loss_G.backward()
            optimizer_G.step()

            # train discriminators, starting with A
            optimizer_D_A.zero_grad()

            # train to discriminate real images as real
            loss_real = criterion_GAN(D_A(real_A), valid)
            # train to discriminate fake images as fake
            loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)

            loss_D_A = (loss_real + loss_fake) / 2.0

            loss_D_A.backward()
            optimizer_D_A.step()

            # then discriminator B
            optimizer_D_B.zero_grad()

            # train to discriminate real images as real
            loss_real = criterion_GAN(D_B(real_B), valid)
            # train to discriminate fake images as fake
            loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2.0
            loss_D_B.backward()
            optimizer_D_B.step()

            # Total Loss
            loss_D = (loss_D_A + loss_D_B) / 2

        print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f - (adv : %f, cycle : %f, identity : %f)]'
               % (epoch + 1, TOTAL_EPOCHS,  # [Epoch -]
                  i + 1, len(dataloader),  # [Batch -]
                  # this is the total of the both discriminators
                  loss_D.item(),  # [D loss -]
                  loss_G.item(),  # [G loss -]
                  loss_GAN.item(),  # [adv -]
                  loss_cycle.item(),  # [cycle -]
                  loss_identity.item())),  # [identity -]
        # save progress as images
        losses.add_loss([loss_D_A.item(), loss_D_B.item(), loss_D.item()],
                        [loss_GAN_AB.item(), loss_GAN_BA.item(), loss_GAN.item()],
                        loss_GAN.item(), loss_cycle.item(), loss_identity.item())
        sample_images(image_folder, epoch, G_AB, G_BA, val_dataloader)
    losses.render_and_save(image_folder)
    save_model(G_AB, image_folder.name)


def test_graph():
    folder_name = get_image_folder()
    losses = TotalLoss()
    losses.add_loss([0.836, 0.601, 0.718], [0.908, 0.806, 0.857], 0.857, 0.381, 0.345)
    losses.add_loss([0.389, 0.387, 0.388], [0.437, 0.458, 0.447], 0.447, 0.354, 0.337)
    losses.add_loss([0.249, 0.332, 0.291], [0.364, 0.464, 0.414], 0.414, 0.241, 0.242)
    losses.add_loss([0.31, 0.252, 0.281], [0.379, 0.363, 0.371], 0.371, 0.226, 0.211)
    losses.add_loss([0.385, 0.367, 0.376], [0.266, 0.452, 0.359], 0.359, 0.339, 0.322)
    losses.add_loss([0.297, 0.287, 0.292], [0.221, 0.26, 0.241], 0.241, 0.173, 0.16)
    losses.add_loss([0.254, 0.329, 0.292], [0.483, 0.355, 0.419], 0.419, 0.478, 0.407)
    losses.add_loss([0.267, 0.335, 0.301], [0.391, 0.292, 0.341], 0.341, 0.266, 0.249)
    losses.add_loss([0.26, 0.318, 0.289], [0.37, 0.451, 0.411], 0.411, 0.219, 0.191)
    losses.add_loss([0.367, 0.403, 0.385], [0.378, 0.336, 0.357], 0.357, 0.204, 0.168)
    losses.render_and_save(folder_name)


if __name__ == '__main__':
    image_cyclegan()

import os
import sys

import numpy as np

from pathlib import Path
from random import shuffle

import torch
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 1
ROOT_FOLDER = Path.cwd() / 'data' / 'Split'
TEST_RATIO = 0.2
MAX_FILES = 1500


class GDDataSet(Dataset):
    def __init__(self, files):
        self.file_list = files
        self.labels = [float(1) if x.startswith('SBD') else float(0) for x in self.file_list]

    def __getitem__(self, index):
        sfx = np.load(ROOT_FOLDER / self.file_list[index])
        # return a tensor, not an array
        data = torch.from_numpy(sfx)
        data = data.cuda()
        return data.float(), torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)


def get_datasets():
    files = os.listdir(ROOT_FOLDER)
    shuffle(files)

    if MAX_FILES > 0:
        files = files[:MAX_FILES]

    sbd_files = []
    aud_files = []
    for i in files:
        if i.startswith('SBD'):
            sbd_files.append(i)
        else:
            aud_files.append(i)
    remove_sbd = int(len(sbd_files) * TEST_RATIO)
    test_files = []
    for i in range(remove_sbd):
        test_files.append(sbd_files.pop(0))

    remove_aud = int(len(aud_files) * TEST_RATIO)
    test_files = []
    for i in range(remove_aud):
        test_files.append(aud_files.pop(0))

    train_files = sbd_files
    train_files.extend(aud_files)

    shuffle(train_files)
    shuffle(test_files)

    train_dataset = GDDataSet(train_files)
    test_dataset = GDDataSet(test_files)

    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


def get_cyclegan_datasets():
    pass


if __name__ == '__main__':
    x_train, x_test = get_datasets()
    print(f'Training songs: {len(x_train)}')
    print(f' Testing songs: {len(x_test)}')
    for x_batch, y_batch in x_train:
        #print(len(x_batch))
        print(y_batch)
        #sys.exit()

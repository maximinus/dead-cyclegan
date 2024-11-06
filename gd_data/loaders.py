import os
import random
import librosa
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 44100
AUDIO_SOURCE_DIR = Path('/home/sparky/code/dead-cyclegan/data/split')

SBD_LABEL = 1.0
AUD_LABEL = 0.0


class AudioData(Dataset):
    def __init__(self, files, normalize_audio=False):
        super().__init__()
        self.all_files = files
        # load all files into RAM
        self.audio = []
        for i in tqdm(self.all_files):
            self.load_file(i, normalize_audio)

    def show_top(self, total):
        files = [self.all_files[x] for x in range(total)]
        print(f'Top #{total}: {files}')

    def load_file(self, filename, normalize_audio):
        signal, audio = librosa.core.load(filename, sr=SAMPLE_RATE, mono=False)
        signal = signal.astype(np.float32)
        # this moves it from -1 to +1
        # for the discriminator, we need from 0 to 1
        # currently it is a sign 16 bit
        signal /= 32768.0
        signal += 0.5

        # reshape the signal
        n_channels = 1 if signal.ndim == 1 else signal.shape[1]
        signal = signal.reshape(n_channels, signal.shape[0])

        # normalize the signal
        if normalize_audio:
            signal /= np.max(np.abs(signal))

        if 'SBD' in str(filename):
            self.audio.append([signal, SBD_LABEL])
        else:
            self.audio.append([signal, AUD_LABEL])

    def __getitem__(self, index):
        data = self.audio[index]
        # return data, then label
        return data[0], data[1]

    def __len__(self):
        return len(self.all_files)


def get_all_wav_files(audio_dir=AUDIO_SOURCE_DIR):
    if not os.path.isdir(audio_dir):
        raise OSError(str(audio_dir))
    wav_files = []
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            full_path = os.path.join(subdir, file)
            if str(full_path).lower().endswith('wav'):
                wav_files.append(full_path)
    return wav_files


def get_fileset(string_end):
    show_dirs = os.listdir(AUDIO_SOURCE_DIR)
    audio_samples = []
    for dir in show_dirs:
        # look for folders ending
        if os.path.isdir(AUDIO_SOURCE_DIR / dir) and str(dir.endswith(string_end)):
            audio_samples.extend(get_all_wav_files(AUDIO_SOURCE_DIR / dir))
    return audio_samples


def get_test_train(test_ratio=0.2, file_usage=1.0, batch_size=32):
    # automatically return the test and train datasets of the audio
    audio_files = get_all_wav_files()
    random.shuffle(audio_files)
    print(f'* Started with {len(audio_files)} audio files')

    if file_usage < 1.0:
        files_to_use = int(len(audio_files) * file_usage)
        audio_files = audio_files[:files_to_use]
        print(f'* Only using {len(audio_files)} audio files')

    train_length = int(len(audio_files) * test_ratio)
    train_data = AudioData(audio_files[train_length:])
    test_data = AudioData(audio_files[:train_length])

    training_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # the length of the loaders must be multiplied by the
    print(f'* Training data: {len(training_data) * batch_size} files')
    print(f'*     Test data: {len(test_data) * batch_size} files')

    return training_data, test_data

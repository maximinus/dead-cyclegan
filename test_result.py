import glob
import soundfile
import torch
import numpy as np
from tqdm import tqdm

from pathlib import Path
from datetime import datetime


# needed for the model load
from cyclegan import ResidualBlock, device


MODEL_DIRECTORY = Path.cwd() / 'models'
SONG_DIRECTORY = Path.cwd() / 'test_result' / 'sample_audio'
OUTPUT_DIRECTORY = Path.cwd() / 'test_result'


def load_model():
    # scan folder for all files
    all_files = glob.glob(str(MODEL_DIRECTORY) + '/*.pth')
    current = datetime.now()
    newest_file = [None, 0]
    for i in all_files:
        filename = i.split('/')[-1]
        times = [int(x) for x in filename[:-4].split('_')]
        this_age = times[3] + (times[2] * 24) + (times[1] * (31 * 24)) + (times[0] * (365 * 31 * 24))
        if this_age > newest_file[1]:
            newest_file = [i, this_age]
    # now we have the newest file
    print(f'* Loading model {newest_file[0]}')
    generator = torch.load(newest_file[0])
    generator.eval()

    # move to CPU for computation; otherwise memory load on GPU is too much
    generator.to('cpu')
    return generator


def stitch_audio(left, right):
    # left and right as numpy arrays
    half_audio = 8192
    new_left = []
    for i in tqdm(range(len(left) - 1)):
        base_audio = [(x * 2.0) - 1.0 for x in left[i + 1]]
        prev_audio = [(x * 2.0) - 1.0 for x in left[i]]
        final_left_audio = []
        for j in range(half_audio):
            final_left_audio.append(base_audio[j] + prev_audio[j + half_audio])
        new_left.extend(final_left_audio)
    new_right = []
    for i in tqdm(range(len(right) - 1)):
        base_audio = [(x * 2.0) - 1.0 for x in right[i + 1]]
        prev_audio = [(x * 2.0) - 1.0 for x in right[i]]
        final_right_audio = []
        for j in range(half_audio):
            final_right_audio.append(base_audio[j] + prev_audio[j + half_audio])
        new_right.extend(final_right_audio)
    full_array = np.array([new_left, new_right])
    full_array = np.transpose(full_array)
    soundfile.write('./test_result/result.wav', full_array, 44100, subtype='PCM_16', format='WAV')


def build_song(model):
    # we need get all the files
    all_files = glob.glob(str(SONG_DIRECTORY) + '/*')
    # sort by channel
    left_channel = []
    right_channel = []
    for i in all_files:
        filename = i.split('/')[-1]
        if filename.startswith('left'):
            left_channel.append(i)
        else:
            right_channel.append(i)
    # now sort by index number
    # filename is of type SBD_00001.wav or similar
    left_channel = sorted(left_channel, key=lambda x: int(x[-9:-4]))
    right_channel = sorted(right_channel, key=lambda x: int(x[-9:-4]))

    # load
    left_arrays = []
    for i in tqdm(left_channel):
        audio = torch.tensor(np.load(i))
        audio = (audio[None, :]).float()
        # move to cuda howto
        # audio = audio.to(device)
        audio = model(audio)
        audio = audio[0].tolist()
        left_arrays.append(audio)

    right_arrays = []
    for i in tqdm(right_channel):
        audio = torch.tensor(np.load(i))
        audio = (audio[None, :]).float()
        audio = model(audio)
        audio = audio[0].tolist()
        right_arrays.append(audio)

    # join together
    print(f'* Stitching {len(left_arrays)} files')
    stitch_audio(left_arrays, right_arrays)


def play_song(song):
    pass


if __name__ == '__main__':
    sound_model = load_model()
    build_song(sound_model)
    #play_song(audio)

#!/usr/bin/env python
import math
import sys

import torch
import random
import shutil
import os.path

import librosa
import argparse
import soundfile

import numpy as np
from tqdm import tqdm
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from cyclegan_with_batchnorm import get_generator, ResidualBlock


INPUT_SAMPLE_RATE = 44100
SAMPLE_SIZE = 16384
HALF_SAMPLE = int(SAMPLE_SIZE / 2)

# call with
# --split path: split up the song and place it in ./song_data/name
# --join path:  join up the files in ./song_data/name_x where X is an index starting at 0


def get_volume_level(x):
    # given a value x, return the volume
    # where x is in range -1 -> +1
    # and get_volume_level(-x) + get_volume_level(+x) = 1
    # it's mirrored, so -ve is same as +ve
    # We use a sine wave from -0.5 PI  to +0.5 PI
    return (math.sin((math.pi * x) / 2.0) + 1.0) / 2.0


def split_track(audio, song_name, left):
    # There needs to be a first item that is id half sample size of 0 to start
    first_element = np.concatenate(([0] * HALF_SAMPLE, audio[:HALF_SAMPLE]))
    audio_split = [first_element]
    audio_length = len(audio)
    index = 0
    while index + SAMPLE_SIZE < audio_length:
        audio_sample = audio[index:index + SAMPLE_SIZE]
        audio_split.append(audio_sample)
        index += HALF_SAMPLE
    extend_size = SAMPLE_SIZE - (audio_length - index)
    last_array = np.concatenate((audio[index:], np.array([0] * extend_size)))
    audio_split.append(last_array)
    return audio_split


def save_song(left_channel, right_channel, song_name):
    # remove the extension from the name
    short_name = song_name.split('.')[0]
    dir_path = f'./song_data/{short_name}'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_base = f'{dir_path}/{short_name}'
    # channels are the same lengths
    print('* Writing audio')
    for index in range(len(left_channel)):
        soundfile.write(f'{file_base}_left_{index}.wav', left_channel[index], 44100, 'PCM_16')
        soundfile.write(f'{file_base}_right_{index}.wav', right_channel[index], 44100, 'PCM_16')


def load_song(dir_path, base_name=None):
    if base_name is None:
        base_name = dir_path.name
    full_path = f'{dir_path}/{base_name}'
    left_channel = []
    right_channel = []
    index = 0
    while True:
        left_file = f'{full_path}_left_{index}.wav'
        right_file = f'{full_path}_left_{index}.wav'
        if os.path.exists(left_file):
            left_channel.append(librosa.load(left_file, sr=INPUT_SAMPLE_RATE, mono=True)[0])
            right_channel.append(librosa.load(right_file, sr=INPUT_SAMPLE_RATE, mono=True)[0])
            index += 1
        else:
            return left_channel, right_channel


def split_song(song_path):
    # split the song into left and right channels
    # for both channels, starting at index -8192 of the channel
    print(f'* Splitting {song_path}')
    audio, sr = librosa.load(song_path, sr=INPUT_SAMPLE_RATE, mono=False)
    if len(audio) != 2:
        print('* Error: File must be stereo')
        return
    print(f'* Loaded file, #{math.ceil(len((audio) / SAMPLE_SIZE) + 1)} samples')
    left_samples = split_track(audio[0], song_path.name, left=True)
    right_samples = split_track(audio[1], song_path.name, left=False)
    save_song(left_samples, right_samples, song_path.name)


def join_channel(channel):
    joined = []
    delta_volume = 1.0 / HALF_SAMPLE
    # we start joining from halfway through; but the first is half empty
    for i in tqdm(range(len(channel) - 1)):
        first = channel[i]
        second = channel[i + 1]
        for audio_index in range(HALF_SAMPLE):
            # last half of the first one and the first half of the last one
            second_volume = delta_volume * HALF_SAMPLE
            first_volume = 1.0 - second_volume
            final_audio = (first[audio_index] * first_volume) + (second[audio_index] * second_volume)
            joined.append(final_audio)
    return joined


def join_song(dir_path, base_name=None):
    # you have a bunch of files for left and right channels, numbered 1...n got both channels
    # we need to join them, 3 at a time; each sample is 16384 but the rendered sound is 8192
    # For example, take sound sample 2
    # This requires: the right 4096 of sample 1; the middle 8192 of sample 2 and the right 4096 of sample 3
    print(f'* Joining song in {dir_path}')
    left, right = load_song(dir_path, base_name)
    left_joined = np.array(join_channel(left))
    right_joined = np.array(join_channel(right))
    # merge and save
    output = np.hstack((left_joined.reshape(-1, 1), right_joined.reshape(-1, 1)))
    filename = f'./song_data/joined/{dir_path.name}_joined.wav'
    soundfile.write(filename, output, 44100, 'PCM_16')


def test_write():
    filename = f'./song_data/joined/test_joined.wav'
    array1 = np.array([random.random() for x in range(10000)])
    array2 = np.array([random.random() for x in range(10000)])
    output = np.hstack((array1.reshape(-1, 1), array2.reshape(-1, 1)))
    soundfile.write(filename, output, 44100, 'PCM_16')


def wav_to_generator(filepath):
    audio, _s = librosa.core.load(filepath, sr=44100, mono=True)

    # change to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32)
        audio /= 32768.

    n_channels = 1
    signal = audio.reshape(n_channels, audio.shape[0])

    # normalize the signal
    # signal /= np.max(np.abs(signal))
    return signal


def generate_sbd(generator, audio):
    # audio needs to a tensor of some kind, not a numpy array
    # actually a tensor([[[data...]]]])
    sbd_as_tensor = torch.tensor([audio])
    sbd_as_tensor = sbd_as_tensor.to(device)
    # Check with what it really is
    new_sbd = generator(sbd_as_tensor).detach().cpu().numpy()
    new_sbd = new_sbd.reshape(new_sbd.shape[2],)
    return new_sbd


def clean_song(dir_path):
    print(f'* Cleaning song in {dir_path}')
    # load the generator in ./saved_models/runs/aud_to_sbd.path
    net_data = torch.load('./models/20_10_15_17.pth')
    generator = get_generator()
    generator.load_state_dict(net_data)
    generator.eval()
    # take the files in the dir_path
    cleaned_audio = []
    for aud_file in tqdm(os.listdir(dir_path)):
        np_audio = wav_to_generator(f'{dir_path}/{aud_file}')
        # push them through the generator
        cleaned_audio.append([generate_sbd(generator, np_audio), Path(aud_file).name])
    # output them to a tmp folder in song_data
    # first clean the tmp folder
    shutil.rmtree('./song_data/cleaned')
    os.mkdir('./song_data/cleaned')
    for i in cleaned_audio:
        audio_data = i[0]
        name = i[1]
        soundfile.write(f'./song_data/cleaned/{name}', audio_data, 44100, 'PCM_16')
    # finally, join the audio
    join_song(Path('./song_data/cleaned'), base_name='brown_eyed_women')


def get_args():
    parser = argparse.ArgumentParser(prog='song editor')
    parser.add_argument('-s', '--split', required=False)
    parser.add_argument('-j', '--join', required=False)
    parser.add_argument('-c', '--clean', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    all_args = sum([0 if x is None else 1 for x in [args.split, args.join, args.clean]])
    if all_args == 0 or all_args > 1:
        print('song_editor\n  --split song_path\n  --join dir_path\n  --clean dir_path')
    elif args.split is not None:
        split_song(Path(args.split))
    elif args.join is not None:
        join_song(Path(args.join))
    elif args.clean is not None:
        clean_song(Path(args.clean))

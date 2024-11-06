import os
import math
import torch
import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from pathlib import Path
from tqdm import tqdm

GD_SHOWS_DIR = Path('/home/sparky/code/dead-cyclegan/data/Original')
MEL_OUTPUT = Path('/home/sparky/data/MLData/GD_MEL')
EXAMPLE_WAV = Path('/home/sparky/code/dead-cyclegan/data/minglewood.wav')
SAMPLE_RATE = 44100
# Size of FFT window
N_FFT = 2048
N_MELS = 512
# the size of the y axis is SLICE_SIZE + 1 // hop_length
SLICE_SIZE = 65535
# Number of samples between successive frames
HOP_LENGTH = 128


def save_image(spectrogram, index):
    fig = plt.figure(figsize=(1, 1), dpi=512)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(spectrogram, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='log')
    # Save the spectrogram as an image
    output_image_path = MEL_OUTPUT / f'{index}_spectrogram.png'
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def convert_song_to_mel(audio_file, directory, file_index):
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
    # get the min/max volume differences for the whole file, so we can normalize on that
    song_mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
                                              hop_length=HOP_LENGTH, n_mels=N_MELS)
    song_db = librosa.power_to_db(song_mel, ref=np.max)
    global_min = np.min(song_db)
    global_max = np.max(song_db)
    total_slices = int(math.floor(len(audio) / SLICE_SIZE))
    index = 0
    for i in range(total_slices):
        audio_to_render = audio[index:index + SLICE_SIZE]
        index += SLICE_SIZE
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_to_render, sr=SAMPLE_RATE, n_fft=N_FFT,
                                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        normalized_spectrogram = (mel_spectrogram_db - global_min) / (global_max - global_min)
        mel_spectrogram_tensor = torch.tensor(normalized_spectrogram, dtype=torch.float32)
        np.save(directory / f'{file_index}_fp32.npy', mel_spectrogram_tensor)
        file_index += 1
    return file_index


def convert_show(show_directory):
    new_directory = MEL_OUTPUT / f'{show_directory.name}'
    os.mkdir(new_directory)
    file_index = 0
    for i in tqdm(os.listdir(show_directory)):
        sound_file = show_directory / i
        file_index = convert_song_to_mel(sound_file, new_directory, file_index)


def convert_all_shows():
    shows = os.listdir(GD_SHOWS_DIR)
    for i in shows:
        full_path = GD_SHOWS_DIR / i
        print(f'* Converting {full_path}')
        convert_show(full_path)


if __name__ == '__main__':
    convert_all_shows()
    #convert_song_to_mel(MEL_OUTPUT / 'test.wav', MEL_OUTPUT, 0)

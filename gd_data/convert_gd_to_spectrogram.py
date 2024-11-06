import math
import os

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
N_FFT = 1023  # Size of FFT window
HOP_LENGTH = 127  # Number of samples between successive frames
SLICE_SIZE = 65536


def base_example():
    # original code to load, transform and transform back
    audio, _ = librosa.load(EXAMPLE_WAV, sr=SAMPLE_RATE)
    index = 0
    slice_size = 65536
    total_slices = int(math.floor(len(audio) / slice_size))

    # size of spectrogram is
    # [1 + N_FFT // 2, audio_length / hop_length + 1]

    audio_sections = []
    for i in tqdm(range(total_slices)):
        audio_to_render = audio[index:index + slice_size]
        index += slice_size
        abs_spectrogram = np.abs(librosa.stft(audio_to_render, n_fft=N_FFT, hop_length=HOP_LENGTH))
        if i < 10:
            save_image(abs_spectrogram, i)
        audio_sections.append(librosa.griffinlim(abs_spectrogram, n_iter=32, n_fft=N_FFT, hop_length=HOP_LENGTH))
    merged_audio = np.concatenate(audio_sections)
    sf.write(MEL_OUTPUT / 'sftf.wav', merged_audio, SAMPLE_RATE, 'PCM_16')



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


def convert_song_to_stft(audio_file, directory, file_index):
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
    total_slices = int(math.floor(len(audio) / SLICE_SIZE))
    index = 0
    for i in range(total_slices):
        audio_to_render = audio[index:index + SLICE_SIZE]
        index += SLICE_SIZE
        abs_spectrogram = np.abs(librosa.stft(audio_to_render, n_fft=N_FFT, hop_length=HOP_LENGTH))
        # convert to float16 and save
        np.save(directory / f'{file_index}_fp16.npy', np.float16(abs_spectrogram))
        file_index += 1
    return file_index


def convert_show(show_directory):
    new_directory = MEL_OUTPUT / f'{show_directory.name}'
    os.mkdir(new_directory)
    file_index = 0
    for i in tqdm(os.listdir(show_directory)):
        sound_file = show_directory / i
        file_index = convert_song_to_stft(sound_file, new_directory, file_index)


def convert_all_shows():
    shows = os.listdir(GD_SHOWS_DIR)
    for i in shows:
        full_path = GD_SHOWS_DIR / i
        print(f'* Converting {full_path}')
        convert_show(full_path)


if __name__ == '__main__':
    convert_all_shows()

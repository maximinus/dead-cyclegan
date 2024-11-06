import os
import shutil

from tqdm import tqdm
from scipy.io import wavfile
import scipy.io.wavfile as wav

ROOT_FOLDER = '/home/sparky/data/code/dead-cyclegan/data/Original'
OUTPUT_FOLDER = '/home/sparky/data/MLData/GD_sliced'
SAMPLE_RATE = 44100
SAMPLE_LENGTH = 65536


def find_directories_with_wavs(directory):
    directories_with_wavs = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                if root not in directories_with_wavs:
                    directories_with_wavs.append(root)
                # Stop checking other files in this directory
                break
    return directories_with_wavs


def get_wav_files():
    all_dirs = find_directories_with_wavs(ROOT_FOLDER)
    file_paths = []
    for wav_dir in all_dirs:
        for file in os.listdir(wav_dir):
            if file.endswith('.wav'):
                file_paths.append(os.path.join(wav_dir, file))
    return file_paths


def save_chunk(data, channel_name, chunk_number):
    chunk_filename = f'{str(chunk_number).zfill(5)}_{channel_name}.wav'
    full_path = f'{OUTPUT_FOLDER}/{chunk_filename}'
    wav.write(full_path, SAMPLE_RATE, data)


def split_and_save_wav(filename, index):
    # Read the stereo WAV file
    sample_rate, audio_data = wavfile.read(filename)

    # Assuming audio_data is a numpy array with two columns: [left, right]
    left_channel = audio_data[:, 0]
    right_channel = audio_data[:, 1]

    # Calculate the number of frames per one second
    frames_per_second = sample_rate

    # Split and save the chunks
    for i in range(0, len(left_channel), SAMPLE_LENGTH):
        left_chunk = left_channel[i:i + SAMPLE_LENGTH]
        right_chunk = right_channel[i:i + SAMPLE_LENGTH]

        # Check if the chunk is less than a second and pad if necessary
        if len(left_chunk) < SAMPLE_LENGTH:
            continue

        label = 'SBD' if 'SBD' in filename else 'AUD'
        save_chunk(left_chunk, f'LEFT_{label}', index)
        save_chunk(right_chunk, f'RIGHT_{label}', index)
        index += 1
    return index


def clear_directory(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  # Remove files and links
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directories


if __name__ == '__main__':
    clear_directory(OUTPUT_FOLDER)
    wav_files = get_wav_files()
    index = 0
    for i in tqdm(wav_files):
        index = split_and_save_wav(i, index)

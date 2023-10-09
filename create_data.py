import os
import random
import shutil
import sys
import math
from pathlib import Path

import numpy
import numpy as np
import soundfile

import librosa as librosa
from tqdm import tqdm


# defaults
FREQUENCY = 44100
# samples per selection - power of 2 makes it easier to convolve
SAMPLE_LENGTH = 16384
# output directories
OUTPUT_FOLDERS = ['Split']

ROOT_FOLDER = Path.cwd() / 'data'
SOURCE_FOLDER = ROOT_FOLDER / 'Original'
OUTPUT_FOLDER = ROOT_FOLDER / 'split'
SAMPLE_FOLDER = Path.cwd() / 'test_result' / 'sample_audio'
SAMPLE_FILE = Path.cwd() / 'test_result' / 'Around_And_Around.wav'
WAV_FOLDER = ROOT_FOLDER / 'wav_split'

DEBUG = False


def error(message):
    print(f'  Error: {message}')
    sys.exit(False)


def check_single_file(input_file, output_folder):
    audio, sr = librosa.load(input_file, sr=44100, mono=False)
    #audio = remove_silence(audio)
    length = SAMPLE_LENGTH
    skip = SAMPLE_LENGTH / 2
    audio_index = 0
    index = 0
    while audio_index + length <= len(audio[0]):
        left_output = output_folder / f'{index}_left_.wav'
        right_output = output_folder / f'{index}_right_.wav'
        # they need to be saved as pure audio for now
        data_left = resample(audio[0][audio_index:audio_index + length])
        data_right = resample(audio[1][audio_index:audio_index + length])
        soundfile.write(left_output, data_left, 44100, subtype='PCM_16', format='WAV')
        soundfile.write(right_output, data_right, 44100, subtype='PCM_16', format='WAV')
        audio_index += skip
        index += 1


def stitch_track():
    # do the left and right channels separately
    left = []
    for i in range(277):
        file = f'{i}_left_.wav'
        audio, sr = librosa.load(file, sr=44100, mono=True)
        left.append(audio)
    full_array = []
    for i in range(276):
        start_audio = left[i].tolist()
        end_audio = left[i + 1].tolist()
        start_index = int(1.5 * 44100)
        end_index = 0
        for j in range(int(1.5 * 44100)):
            volume1 = start_audio[start_index]
            volume2 = end_audio[end_index]
            volume1 = (volume1 * 2.0) - 1.0
            volume2 = (volume2 * 2.0) - 1.0
            full_array.append(volume1 + volume2)
            start_index += 1
            end_index += 1
    full_array = numpy.array(full_array)
    soundfile.write('resampled.wav', full_array, 44100, subtype='PCM_16', format='WAV')


def get_all_wavs(source):
    if not os.path.isdir(source):
        error(f'Input path {input} does not exist')
    # get all folders here
    folders = []
    total = 0

    print([x for x in filter(lambda x: (os.path.isdir(source / x)), os.listdir(source))])

    for folder in filter(lambda x: (os.path.isdir(source / x)), os.listdir(source)):
        all_wav_files = [x for x in filter(lambda x: (str(x).endswith('wav')), os.listdir(source / folder))]
        folders.append([folder, all_wav_files])
        total += len(all_wav_files)
    print(f'  * Found {total} files in {len(folders)} folders')
    return folders


def remove_silence(audio):
    # this changes audio and returns the trimmed part
    librosa.effects.trim(audio, top_db=10)
    return audio


def save_single_channel(audio, index, sbd=True, left=True, sample=False):
    length = SAMPLE_LENGTH
    audio_index = 0
    inner_index = 0
    channel_name = 'left' if left is True else 'right'
    while audio_index + length <= len(audio):
        if sbd:
            if not sample:
                output_filename = OUTPUT_FOLDER / f'SBD_{index}_{inner_index}_{channel_name}.npy'
            else:
                output_filename = SAMPLE_FOLDER / f'{channel_name}_{inner_index:05d}.npy'
        else:
            if not sample:
                output_filename = OUTPUT_FOLDER / f'AUD_{index}_{inner_index}_{channel_name}.npy'
            else:
                output_filename = SAMPLE_FOLDER / f'{channel_name}_{inner_index:05d}.npy'
        # they need to be saved as pure audio for now
        audio_slice = resample(audio[audio_index:audio_index + length])
        # this is a numpy array, so save it as that
        np.save(str(output_filename), np.float32(audio_slice))
        audio_index += int(SAMPLE_LENGTH / 2)
        inner_index += 1
    # now we need to do the last one, which won't be long enough
    # make sure the length is not zero though
    resample_data = audio[audio_index:]
    if len(resample_data) <= 0:
        # nothing to do
        return
    final_audio = resample(resample_data)
    missing_data = numpy.array([0.0] * (length - len(final_audio)))
    final_numpy = np.concatenate((final_audio, missing_data))
    if len(final_numpy) != length:
        raise ValueError(f'Wrong size: {len(final_numpy)}')
    if sbd:
        if not sample:
            output_filename = OUTPUT_FOLDER / f'SBD_{index}_{inner_index}_{channel_name}.npy'
        else:
            output_filename = SAMPLE_FOLDER / f'{channel_name}_{inner_index:05d}.npy'
    else:
        if not sample:
            output_filename = OUTPUT_FOLDER / f'AUD_{index}_{inner_index}_{channel_name}.npy'
        else:
            output_filename = SAMPLE_FOLDER / f'{channel_name}_{inner_index:05d}.npy'

    # change to 32 bit floats, else 8 bytes (!) per value
    np.save(str(output_filename), np.float32(final_numpy))


def save_output(audio, is_sbd, index):
    save_single_channel(audio[0], index, sbd=is_sbd, left=True)
    save_single_channel(audio[1], index, sbd=is_sbd, left=False)


def process_files(wav_files):
    for index, f in enumerate(tqdm(wav_files)):
        audio, sr = librosa.load(f, sr=FREQUENCY, mono=False)
        # aud or sbd? check parent
        is_sbd_clip = f.parent.name.endswith('SBD')
        save_output(audio, is_sbd_clip, index)


def get_volume_level(x):
    # where x is in range -1 -> +1
    if x > 0.0:
        x *= -1.0
    x += 0.5
    x *= (math.pi * 2.0)
    return min((math.tanh(x) + 1.0) / 2.0, 1.0)


def resample(audio):
    # This gives results in the range -1 -> 1, but we need 0 -> 1
    audio_length = len(audio)
    index = -1.0
    audio_delta = 2.0 / audio_length
    new_audio = []
    for i in audio.tolist():
        # add this line to get volume curvature
        #new_volume = get_volume_level(index) * i
        new_volume = i
        # this is now from -1 to +1
        new_volume = (new_volume + 1.0) / 2.0
        new_audio.append(new_volume)
        index += audio_delta
    return numpy.array(new_audio)


def write_test_sample():
    audio, sr = librosa.load(SAMPLE_FILE, sr=FREQUENCY, mono=False)
    # audio channel, index, sbd/aud (ignored), Left, Sample
    print('* Writing left channel')
    save_single_channel(audio[0], 0, True, True, True)
    print('* Writing right channel')
    save_single_channel(audio[1], 0, True, False, True)


def save_single_channel_wav(output, audio, index, channel_name, sbd=True):
    length = SAMPLE_LENGTH
    audio_index = 0
    inner_index = 0
    while audio_index + length <= len(audio):
        if sbd:
            output_filename = output / f'SBD_{index}_{inner_index}_{channel_name}.wav'
        else:
            output_filename = output / f'AUD_{index}_{inner_index}_{channel_name}.wav'
        # they need to be saved as pure audio for now
        audio_slice = audio[audio_index:audio_index + length]
        soundfile.write(str(output_filename), audio_slice, FREQUENCY, 'PCM_16')
        audio_index += SAMPLE_LENGTH
        inner_index += 1


def write_to_tmp(folder):
    wav_files = get_all_wavs(SOURCE_FOLDER)
    for index, f in enumerate(tqdm(wav_files)):
        audio, sr = librosa.load(f, sr=FREQUENCY, mono=False)
        # aud or sbd? check parent
        is_sbd_clip = f.parent.name.endswith('SBD')
        save_single_channel_wav(folder, audio[0], index, 'left', sbd=is_sbd_clip)
        save_single_channel_wav(folder, audio[1], index, 'right', sbd=is_sbd_clip)


def move_file(source, destination):
    if DEBUG is True:
        print(f'Moving {source} to {destination}')
    else:
        os.rename(source, destination)


def split_wavs():
    # split all files into a tmp folder
    tmp_folder = WAV_FOLDER / 'tmp'
    #write_to_tmp(tmp_folder)
    # get a list of all the files
    #all_files = get_all_wav_files(tmp_folder)
    all_files = []
    print(f'  * Got {len(all_files)} files')
    print(all_files[0])
    # split into sbd and aud
    aud = []
    sbd = []
    for i in all_files:
        if str(i).split('/')[-1].startswith('SBD'):
            sbd.append(i)
        else:
            aud.append(i)
    # shuffle
    random.shuffle(aud)
    random.shuffle(sbd)
    print(f'  * {len(sbd)} SBD files')
    print(f'  * {len(sbd)} AUD files')
    # split into test, train sets
    train_sbd_length = int(len(aud) * 0.1)
    train_aud_length = int(len(sbd) * 0.1)
    print(f'  * Moving {len(sbd[train_sbd_length:])} files to X')
    for i in tqdm(sbd[train_sbd_length:]):
        move_file(i, WAV_FOLDER / 'X' / str(i).split('/')[-1])
    print(f'  * Moving {len(sbd[:train_sbd_length])} files to test_X')
    for i in tqdm(sbd[:train_sbd_length]):
        move_file(i, WAV_FOLDER / 'test_X' / str(i).split('/')[-1])
    print(f'  * Moving {len(sbd[train_aud_length:])} file to Y')
    for i in tqdm(aud[train_aud_length:]):
        move_file(i, WAV_FOLDER / 'Y' / str(i).split('/')[-1])
    print(f'  * Moving {len(sbd[:train_aud_length:])} files to test_Y')
    for i in tqdm(aud[:train_aud_length]):
        move_file(i, WAV_FOLDER / 'test_Y' / str(i).split('/')[-1])


def save_wav_single_channel(audio, index, is_sbd, left):
    audio_index = 0
    inner_index = 0
    if left:
        channel_name = 'left'
    else:
        channel_name = 'right'
    while audio_index + SAMPLE_LENGTH <= len(audio):
        if is_sbd:
            output_filename = WAV_FOLDER / f'SBD_{index}_{inner_index}_{channel_name}.wav'
        else:
            output_filename = WAV_FOLDER / f'AUD_{index}_{inner_index}_{channel_name}.wav'
        # they need to be saved as pure audio for now
        audio_slice = audio[audio_index:audio_index + SAMPLE_LENGTH]
        soundfile.write(str(output_filename), audio_slice, FREQUENCY, 'PCM_16')
        audio_index += SAMPLE_LENGTH
        inner_index += 1


def split_wav(is_sbd, audio, index):
    save_wav_single_channel(audio[0], index, is_sbd, True)
    save_wav_single_channel(audio[1], index, is_sbd, False)


def wav_split(files):
    # call this to write the raw audio files in pieces
    index = 0
    for index, f in enumerate(tqdm(files)):
        audio, sr = librosa.load(f, sr=FREQUENCY, mono=False)
        # aud or sbd? check parent
        is_sbd_clip = f.parent.name.endswith('SBD')
        split_wav(is_sbd_clip, audio, index)
        index += 1


def clear_old_data():
    shutil.rmtree(OUTPUT_FOLDER)
    os.mkdir(OUTPUT_FOLDER)


def save_channel(source, destination, channel_name, is_sbd):
    audio_index = 0
    file_index = 0
    while audio_index + SAMPLE_LENGTH < len(source):
        filename = destination / f'{file_index:04}_{channel_name}.wav'
        audio_slice = source[audio_index:audio_index + SAMPLE_LENGTH]
        soundfile.write(str(filename), audio_slice, FREQUENCY, 'PCM_16')
        audio_index += SAMPLE_LENGTH
        file_index += 1


def split_single_wav(directory, file):
    main_dir = OUTPUT_FOLDER / directory.name
    # make sure the directory exists
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    # make the new directory, minus the .wav
    song_dir = main_dir / file[:-4]
    os.mkdir(song_dir)
    is_sbd = str(directory.name).endswith('SBD')
    # load the audio
    audio, sr = librosa.load(SOURCE_FOLDER / directory.name / file, sr=FREQUENCY, mono=False)
    # remove any silence from the wav
    audio = librosa.effects.trim(audio, top_db=20)
    # split into channels and save - trim changes the dimensions
    save_channel(audio[0][0], song_dir, 'LEFT', is_sbd)
    save_channel(audio[0][1], song_dir, 'LEFT', is_sbd)


def split_wav_files(folders):
    for folder in folders:
        new_directory = OUTPUT_FOLDER / folder[0]
        os.mkdir(SOURCE_FOLDER / new_directory)
        for track in tqdm(folder[1]):
            split_single_wav(new_directory, track)


if __name__ == '__main__':
    #all_files = get_all_files(SOURCE_FOLDER)
    #wav_split(all_files)
    #process_files(all_files)
    #write_test_sample()
    wav_folders = get_all_wavs(SOURCE_FOLDER)
    clear_old_data()
    split_wav_files(wav_folders)

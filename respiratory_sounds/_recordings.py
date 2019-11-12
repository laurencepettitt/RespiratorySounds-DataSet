import os
import urllib
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

import librosa
import pandas as pd

from respiratory_sounds import _tools

RecordingMetaInfo = namedtuple('RecordingMetaInfo',
                               ("patient_number", "recording_index", "chest_location", "num_channels",
                                "recording_equipment"))


def _get_data_set_audio_path():
    return os.path.join(_tools.get_temporary_dir(), 'ICBHI_final_database')


def _get_data_set_url():
    return 'https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip'


def _get_data_set_from_url(data_set_audio_path, data_set_url):
    # Download data set to temp file
    data_set_url_parsed = urlparse(data_set_url)
    temp_data_set_file = os.path.join(_tools.get_temporary_dir(), os.path.basename(data_set_url_parsed.path))
    data_set_audio_path_dirname, data_set_audio_path_basename = os.path.split(data_set_audio_path)

    if not os.path.exists(temp_data_set_file):
        print('Downloading the data set zip file from url. This could take a while..')
        urlretrieve(data_set_url, temp_data_set_file)
    else:
        print('Found cached data set zip file.')

    # Unzip data set to directory
    print('Unzipping the data set zip file. This could take a while..')
    with zipfile.ZipFile(temp_data_set_file, 'r') as zip_ref:
        members = zip_ref.namelist()
        desired_member = data_set_audio_path_basename + os.path.sep
        assert desired_member in members
        zip_ref.extractall(path=data_set_audio_path_dirname)

    # Delete temp zip file
    # os.remove(temp_data_set_file)


def _ensure_data_set_available():
    data_set_audio_path = _get_data_set_audio_path()
    if not os.path.isdir(data_set_audio_path):
        _get_data_set_from_url(data_set_audio_path, _get_data_set_url())


def _is_wav(filename: str) -> bool:
    """
    Checks if file filename is apparently a WAVE file by checking for ".wav" at end of string

    Args:
        filename: string of file name e.g. layla.wav

    Returns:
        True if file filename is wav, false otherwise

    """
    return filename.lower().endswith(('.wav', '.wave'))


def _get_wav_dir_entries() -> List[os.DirEntry]:
    """
    Gets list of DirEntry files in dataset_audio_path which are wav files

    Returns:
        List of DirEntry files

    """
    _ensure_data_set_available()
    with os.scandir(_get_data_set_audio_path()) as entries:
        return [entry for entry in entries if _is_wav(entry.name)]


def _extract_file_name_without_extension_from_path(file_path: str) -> str:
    """
    Extracts the "naked" file name i.e without path or file type extension

    Extracts the last part of path and removes the file type (e.g ".wav") from the end

    Args:
        file_path: String of file path

    Returns:
        String of file name without file type at end

    """
    _, file_name = os.path.split(file_path)  # get last part of path (i.e. file name)
    assert file_name != ""
    file_name_without_extension = file_name.split('.')
    assert (len(file_name_without_extension) >= 2)
    return ".".join(file_name_without_extension[:-1])  # join all but last element into a string, delimited by periods


def _get_file_paths(dir_entries):
    """
    Takes a list of files as DirEntry objects and returns the file paths

    Args:
        dir_entries: List of DirEntry objects

    Returns:
        List of file paths as strings

    """
    return [entry.path for entry in dir_entries]


def _load_recording_datum(file_path: str) -> Tuple[Any, Any]:
    """
    Load an audio file from file_path as a floating point time series

    Args:
        file_path: Path to sound file

    Returns:
         Tuple containing:
            audio: np.ndarray [shape=(n,) or (2, n)]
                audio time series

            sample_rate: number > 0 [scalar]
                sampling rate of `audio`
    """
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    return audio, sample_rate


def _load_recording_data(wav_file_paths):
    """
    Get all recordings in data set as floating point time series and sample_rates.

    Each recording is a tuple of a floating point time series and the sample_rate.

    Returns:
        List of tuples containing:
            audio: np.ndarray [shape=(n,) or (2, n)]
                audio time series

            sample_rate: number > 0 [scalar]
                sampling rate of `audio`

    """
    return pd.DataFrame(
        [_load_recording_datum(file_path) for file_path in wav_file_paths],
        columns=("audio_recording", "sample_rate")
    )


def _extract_meta_info_from_file_name(file_name: str):
    """
    Extract information about a recording from it's name.

    Files of recordings are named in a structured way outlined in 'filename_format.txt'
    in _tools.get_data_set_base_path() but also pasted below.

    This means the name gives us information about the recording.

    Elements contained in the filenames:
        Patient number: (101,102,...,226)
        Recording index
        Chest location: (Trachea (Tc), {Anterior (A), Posterior (P), Lateral (L)}{left (l), right (r)})
        Acquisition mode: (sequential/single channel (sc), simultaneous/multichannel (mc))
        Recording equipment:
            (
                AKG C417L Microphone,
                3M Littmann Classic II SE Stethoscope,
                3M Litmmann 3200 Electronic Stethoscope,
                WelchAllyn Meditron Master Elite Electronic Stethoscope
            )

    Args:
        file_name: String name of file containing a recording

    Returns:
        namedtuple('Record',
                    ("patient_number", "recording_index", "chest_location", "num_channels", "recording_equipment"))
    """
    record_data_values = file_name.split('_')
    # let sc (single-channel) be represented by an integer 1 and mc (multi-channel) be represented by an integer 2
    record_data_values[3] = int(1 if record_data_values[3] == "sc" else 2)
    assert (len(record_data_values) == 5)  # for explanation, see docstring
    record_data_values[0] = int(record_data_values[0])  # type cast patient_number into int
    return RecordingMetaInfo(*record_data_values)  # use splat operator to unpack list to arguments


def _load_recording_meta_info(wav_file_paths):
    """
    Extracts meta information for all recordings.

    For each recording sample, there is the following meta information available:
        Patient number (101,102,...,226)
        Recording index
        Chest location (Trachea (Tc), {Anterior (A), Posterior (P), Lateral (L)}{left (l), right (r)})
        Acquisition mode (sequential/single channel (sc), simultaneous/multichannel (mc))
        Recording equipment (
                AKG C417L Microphone,
                3M Littmann Classic II SE Stethoscope,
                3M Litmmann 3200 Electronic Stethoscope,
                WelchAllyn Meditron Master Elite Electronic Stethoscope
            )

    Returns:
        DataFrame with columns:
            ("file_path", "patient_number", "recording_index", "chest_location", "num_channels", "recording_equipment")
    """
    return pd.DataFrame(
        [
            _extract_meta_info_from_file_name(_extract_file_name_without_extension_from_path(wav_file_path))
            for wav_file_path in wav_file_paths
        ]
    )


def load_recordings():
    wav_file_paths = _get_file_paths(_get_wav_dir_entries())
    recording_data = _load_recording_data(wav_file_paths)
    recording_meta_info = _load_recording_meta_info(wav_file_paths)
    num_recordings = len(recording_data)
    assert num_recordings == len(recording_data) == len(recording_meta_info)
    recording_id = pd.Series(range(num_recordings), name="recording_id")
    return pd.concat([recording_id, recording_meta_info, recording_data], axis=1)


if __name__ == '__main__':
    _get_wav_dir_entries()
import os

import pandas as pd

import respiratory_sounds._patients
import respiratory_sounds._recordings
import respiratory_sounds._tools


def convert_diagnosis_class_to_name(label: int) -> str:
    """
    Converts class label into diagnosis name

    Args:
        label: class label int

    Returns:
        String diagnosis

    """
    return _tools.convert_diagnosis_class_to_name(label)


def convert_diagnosis_name_to_class(diagnosis: str) -> int:
    """
    Converts diagnosis name into class label, an int from range(8)

    Args:
        diagnosis: String name of diagnosis

    Returns:
        Integer label of class

    """
    return _tools.convert_diagnosis_name_to_class(diagnosis)


def get_diagnosis_names():
    return _tools.diagnosis_class_name_mapping()


def num_classes():
    return len(get_diagnosis_names())


def get_diagnosis_classes():
    return range(num_classes())


def load_data_frame_with_cache(data_frame_loader, cache_path):
    cache_path_head = os.path.dirname(cache_path)
    if os.path.isfile(cache_path):
        return pd.read_pickle(cache_path)
    else:
        recording_data = data_frame_loader()
        _tools.create_path_if_nonexistent(cache_path_head)
        pd.to_pickle(recording_data, cache_path)  # cache for later
        return recording_data


def load_recordings():
    return _recordings


def load_patients():
    return _patients


def get_recordings_patients_join():
    join_field = "patient_number"
    return pd.merge(left=_recordings, right=_patients, left_on=join_field, right_on=join_field)


def empty_cache_recordings():
    os.remove(_cache_path_recordings)


def empty_cache_patients():
    os.remove(_cache_path_patients)


_cache_path_recordings = os.path.join(_tools.get_temporary_dir(), '_all_recording_data.pkl')
_cache_path_patients = os.path.join(_tools.get_temporary_dir(), '_all_patient_data.pkl')

_recordings = load_data_frame_with_cache(_recordings.load_recordings, _cache_path_recordings)
_patients = load_data_frame_with_cache(_patients.load_patients, _cache_path_patients)

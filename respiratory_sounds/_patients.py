import os
from collections import Mapping, namedtuple

import pandas as pd

from respiratory_sounds import _tools

Patient = namedtuple('Patient',
                     ("patient_number", "age", "sex", "adult_bmi", "child_weight", "child_height"))


def _get_demographic_info_file_path():
    return os.path.join(_tools.get_data_set_base_path(), 'demographic_info.txt')


def _get_patient_diagnoses_csv_path():
    return os.path.join(_tools.get_data_set_base_path(), "patient_diagnosis.csv")


def _load_demographic_information():
    """
    Get demographic information about all patients.

    Data is derived from '_demographic_info_file_path' which contains the following space separated columns:

      - Patient number
      - Age
      - Sex
      - Adult BMI (kg/m2)
      - Child Weight (kg)
      - Child Height (cm)

    Returns:
        Dictionary of patient_number: Patient
            where Patient = namedtuple('Patient',
                                        ("patient_number", "age", "sex", "adult_bmi", "child_weight", "adult_height"))
    """
    patient_info = []
    f = open(_get_demographic_info_file_path())
    for line in f:
        fields = line.strip().split()
        assert len(fields) == 6
        patient_info.append(Patient(*fields))  # uses splat operator to unpack 'fields' list to arguments
    f.close()
    df = pd.DataFrame(patient_info)
    df['patient_number'] = pd.to_numeric(df['patient_number'])
    return df


def _load_patient_diagnoses():
    """
    Gets patient diagnoses into Panda Dataframe

    Returns:
        Panda Data frame of patient diagnoses
    """
    patient_diagnoses = pd.read_csv(
        _get_patient_diagnoses_csv_path(), header=None, names=["patient_number", "diagnosis_class"]
    )
    patient_diagnoses['diagnosis_class'] = patient_diagnoses['diagnosis_class'].map(
        _tools.convert_diagnosis_name_to_class)
    return patient_diagnoses


def load_patients():
    join_field = "patient_number"
    patient_demographics = _load_demographic_information()
    patient_diagnoses = _load_patient_diagnoses()
    return pd.merge(left=patient_diagnoses, right=patient_demographics, left_on=join_field,
                    right_on=join_field).set_index(join_field)

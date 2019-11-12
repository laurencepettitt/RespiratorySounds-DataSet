import os


def get_data_set_base_path():
    return os.path.dirname(__file__)


def get_temporary_dir():
    temp_path = os.path.join(get_data_set_base_path(), 'temp')
    create_path_if_nonexistent(temp_path)
    return temp_path


def create_path_if_nonexistent(path):
    """
    Ensures existence of a file path by creating it if it doesn't exist

    Args:
        path: file path to ensure existence of

    """
    if not os.path.exists(path):
        os.mkdir(path)


def diagnosis_class_name_mapping():
    """
    Defines the one-to-one mapping between class labels and the diagnosis name

    Returns:
        List, defining mapping between class labels and the diagnosis name
    """
    return [
        "Healthy",
        "Asthma",
        "Bronchiectasis",
        "Bronchiolitis",
        "COPD",
        "LRTI",
        "Pneumonia",
        "URTI"
    ]


def convert_diagnosis_class_to_name(label: int) -> str:
    """
    Converts class label into diagnosis name

    Args:
        label: class label int

    Returns:
        String diagnosis

    """
    return diagnosis_class_name_mapping()[label]


def convert_diagnosis_name_to_class(diagnosis: str) -> int:
    """
    Converts diagnosis name into class label, an int from range(8)

    Args:
        diagnosis: String name of diagnosis

    Returns:
        Integer label of class

    """
    return diagnosis_class_name_mapping().index(diagnosis)

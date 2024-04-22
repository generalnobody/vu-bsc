# Handles outputting results to files

from scipy.io import mmwrite
from scipy.sparse import *


def write_mm_file(filepath, data):
    if not filepath.endswith(".mtx"):
        print("error: not a .mtx file")
        return None

    try:
        mmwrite(filepath, data, comment="Results computed by the program")

    except Exception as e:
        print("error: ", e)
        return None

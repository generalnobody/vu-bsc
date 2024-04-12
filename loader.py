# Script responsible for loading the specified sparse matrix into the desired format
from scipy.io import mmread
from scipy.sparse import *


# Checks if the provided file is a MatrixMarket file
def is_mm_format(file_path):
    try:
        with open(file_path, "r") as file:
            line = file.readline()

        print(line)

        if line.startswith("%%MatrixMarket") and "matrix" in line \
                and "coordinate" in line:
            return True
        else:
            print("File not MatrixMarket")
            return False
    except Exception as e:
        print("Error: ", e)
        return False


# Loads the file into one of the chosen sparse matrix formats
def load_mm_file(file_path, fmt):
    try:
        if not is_mm_format(file_path):
            return None

        sparse_matrix = mmread(file_path)

        if fmt == "coo":
            return coo_matrix(sparse_matrix)
        elif fmt == "csr":
            return csr_matrix(sparse_matrix)
        elif fmt == "csc":
            return csc_matrix(sparse_matrix)
        elif fmt == "dia":
            return dia_matrix(sparse_matrix)
        elif fmt == "bsr":
            return bsr_matrix(sparse_matrix)
        elif fmt == "lil":
            return lil_matrix(sparse_matrix)
        elif fmt == "dok":
            return dok_matrix(sparse_matrix)
        else:
            print("Error: unknown format '{}'".format(fmt))
            return None

    except Exception as e:
        return None

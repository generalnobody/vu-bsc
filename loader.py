# Script responsible for loading the specified sparse matrix into the desired format
from scipy.io import mmread
from scipy.sparse import *
import warnings
import torch.sparse


# Checks if the provided file is a MatrixMarket file
def is_mm_format(file_path):
    try:
        with open(file_path, "r") as file:
            line = file.readline()
            file.close()

        if line.startswith("%%MatrixMarket") and "matrix" in line \
                and "coordinate" in line:
            return True
        else:
            print("File not MatrixMarket")
            return False
    except Exception as e:
        print("Error: ", e)
        return False


# Loads the file into one of the chosen Sparse Matrix formats
def load_mm_file(file_path, fmt, pytorch):
    try:
        if not is_mm_format(file_path):
            return None

        sparse_matrix = mmread(file_path)

        return_matrix = None

        # Load matrix into chosen format, SciPy implementation
        if fmt == "coo":
            return_matrix = coo_matrix(sparse_matrix)
        elif fmt == "csr":
            return_matrix = csr_matrix(sparse_matrix)
        elif fmt == "csc":
            return_matrix = csc_matrix(sparse_matrix)
        elif fmt == "dia":
            # Filter out the SparseEfficiencyWarning, which is emitted when loading into DIA with a non-diagonal matrix
            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            return_matrix = dia_matrix(sparse_matrix)
        elif fmt == "bsr":
            return_matrix = bsr_matrix(sparse_matrix)
        elif fmt == "lil":
            return_matrix = lil_matrix(sparse_matrix)
        elif fmt == "dok":
            return_matrix = dok_matrix(sparse_matrix)
        else:
            print("Error: unknown format '{}'".format(fmt))

        # If PyTorch used, change matrix to PyTorch matrix
        if pytorch and return_matrix is not None:
            dense_matrix = return_matrix.toarray()
            torch_matrix = torch.tensor(dense_matrix, dtype=torch.float64)
            if fmt == "coo":
                return_matrix = torch_matrix.to_sparse_coo()
            elif fmt == "csr":
                return_matrix = torch_matrix.to_sparse_csr()
            elif fmt == "csc":
                return_matrix = torch_matrix.to_sparse_csr()
            elif fmt == "bsr":
                blocksize = return_matrix.blocksize
                return_matrix = torch_matrix.to_sparse_bsr(blocksize)
            else:
                return None

        return return_matrix

    except Exception as e:
        return None

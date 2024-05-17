# Measure memory usage of sparse matrices

import objsize
import json
import argparse
from loader import *

with open("./dicts.json", "r") as read_file:
    dicts = json.load(read_file)

format_options = list(dicts['formats_dict'].keys())[:-1]

# With the provided matrix, calculate the space used with each format. Print those. Use different matrices to accomodate for diagonals and dense submatrices

parser = argparse.ArgumentParser(description="calculates the number of bytes used by each format with the provided matrix")

parser.add_argument("-i", "--input", help="input file, MatrixMarket format", required=True)

args = parser.parse_args()

temp_mtx = load_mm_file(args.input, 'coo')
nnz = temp_mtx.nnz
entry_type = temp_mtx.dtype
entry_size = entry_type.itemsize
print(f"Number of non-zero entries in matrix is {nnz}. The type is {entry_type} with size {entry_size} bytes.\n"
      f"The non-zero entries require {nnz * entry_size} bytes.")

for fmt in format_options:
    mtx = load_mm_file(args.input, fmt)
    size = objsize.get_deep_size(mtx)
    print(f"{fmt.upper()} - {size} bytes")


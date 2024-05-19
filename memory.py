# Measure memory usage of sparse matrices and compare to theoretical requirements

import objsize
import json
import argparse
from loader import *

with open("./dicts.json", "r") as read_file:
    dicts = json.load(read_file)

format_options = list(dicts['formats_dict'].keys())[:-1]

parser = argparse.ArgumentParser(description="calculates the number of bytes used by each format with the provided matrix")

parser.add_argument("-i", "--input", help="input file, MatrixMarket format", required=True)

args = parser.parse_args()

temp_mtx = load_mm_file(args.input, 'coo')
nnz = temp_mtx.nnz
entry_type = temp_mtx.dtype
entry_size = entry_type.itemsize
print(f"Number of non-zero entries in matrix is {nnz}. The type is {entry_type} with size {entry_size} bytes.\n"
      f"The non-zero entries require {nnz * entry_size} bytes.")

print("\nTheoretical memory usage per format (explanation):")
print("COO - stores triplets (i,j,v) where i and j are int32 (4 bytes) and v is float64 (8 bytes)")
print("CSR - stores row pointers, column indices, and values. Requires storage for row pointers (number of rows + 1) (int32, to 4 bytes per entry), column indices (int32, to 4 bytes per entry), values")
print("CSC - same as CSR, but for columns")
print("DIA - stores data for diagonals, if lot of bytes, suggests many diagonals or inefficient diagonals")
print("BSR - stores blocks of values, size depends on block size or density pattern")
print("LIL - stores each row as a list. Size depends on how efficient construction is but is generally close to theoretical number of bytes required + a bit of overhead")
print("DOK - stores everything in a dictionary, using indices as keys and values as values. flexible but usually high overhead")

# print("\nTheoretical memory usage per format (calculation):")
# print(f"COO - {nnz * 4 * 2 + nnz * 8} bytes")
# print(f"CSR - {(num_rows + 1) * 4 + nnz * 4 + nnz * 8} bytes")
# print(f"CSC - {(num_cols + 1) * nnz * 4 + nnz * 8} bytes")
# print(f"DIA - {num_diagonals * 4 + num_diagonals * num_rows * 8} bytes")
# print(f"BSR - {num_blocks * (block_size**2) * 8 + num_blocks * 4 + (num_rows / block_size + 1) * 4} bytes")
# print(f"LIL - {nnz * 4 + nnz * 8 + num_rows * 4} bytes")
# print(f"DOK - {nnz * 2 * 4 + nnz * 8} bytes (excluding dict structure overhead)")

num_rows = temp_mtx.shape[0]
num_cols = temp_mtx.shape[1]
print("\nMemory usage:")
results = [["Format", "Theoretical Size (bytes)", "Actual Size (bytes)"]]
for fmt in format_options:
    mtx = load_mm_file(args.input, fmt)

    new_result = [fmt.upper()]
    theoretical_size = -1
    if fmt == "coo":
        theoretical_size = nnz * 4 * 2 + nnz * 8
    elif fmt == "csr":
        theoretical_size = (num_rows + 1) * 4 + nnz * 4 + nnz * 8
    elif fmt == "csc":
        theoretical_size = (num_cols + 1) * nnz * 4 + nnz * 8
    elif fmt == "dia":
        num_diagonals = mtx.data.shape[0]
        theoretical_size = num_diagonals * 4 + num_diagonals * num_rows * 8
    elif fmt == "bsr":
        num_blocks = mtx.indices.size
        block_size = mtx.blocksize
        theoretical_size = num_blocks * (block_size[0] * block_size[1]) * 8 + num_blocks * 4 + (int(num_rows / block_size[0]) + 1) * 4
    elif fmt == "lil":
        theoretical_size = nnz * 4 + nnz * 8 + num_rows * 4
    elif fmt == "dok":
        theoretical_size = nnz * 2 * 4 + nnz * 8  # This excludes dict structure overhead
    new_result.append(str(theoretical_size))

    actual_size = objsize.get_deep_size(mtx)
    new_result.append(str(actual_size))

    results.append(new_result)

col_widths = [max(len(item) for item in col) for col in zip(*results)]
for res in results:
    print("    ".join(f"{item.ljust(width)}" for item, width in zip(res, col_widths)))




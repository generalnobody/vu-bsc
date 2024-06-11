# Measure memory usage of sparse matrices and compare to theoretical requirements

from pympler import asizeof
import json
import argparse
import numpy as np
from loader import *


try:
    with open("./dicts.json", "r") as read_file:
        dicts = json.load(read_file)

        format_options = list(dicts['formats_dict'].keys())[:-1]
except Exception as e:
    print(e)
    exit(1)

parser = argparse.ArgumentParser(description="calculates the number of bytes used by each format with the provided matrix (does not support pytorch library)")

parser.add_argument("-i", "--input", help="input file, MatrixMarket format", required=True)
parser.add_argument('-o', '--output', help="CSV file to output result to; if not specified, only prints result to stdout (optional)")

args = parser.parse_args()

if args.output is not None and not args.output.endswith(".csv"):
    parser.error("input file format should be MatrixMarket, with .mtx extension")

try:
    temp_mtx = load_mm_file(args.input, 'coo', False)
    if temp_mtx is None:
        parser.error("unknown input file format")
    nnz = temp_mtx.nnz
    entry_type = temp_mtx.dtype
    entry_size = entry_type.itemsize
    base_bytes = nnz * entry_size
    print(f"Number of non-zero entries in matrix is {nnz}. The type is {entry_type} with size {entry_size} bytes.\n"
          f"The non-zero entries require {base_bytes} bytes.")

    print("\nTheoretical memory usage per format (explanation):")
    print("COO - stores triplets (i,j,v) where i and j are int32 (4 bytes) and v is float64 (8 bytes)")
    print(
        "CSR - stores row pointers, column indices, and values. Requires storage for row pointers (number of rows + 1) (int32, to 4 bytes per entry), column indices (int32, to 4 bytes per entry), values")
    print("CSC - same as CSR, but for columns")
    print("DIA - stores data for diagonals, if lot of bytes, suggests many diagonals or inefficient diagonals")
    print("BSR - stores blocks of values, size depends on block size or density pattern")
    print(
        "LIL - stores each row as a list. Size depends on how efficient construction is but is generally close to theoretical number of bytes required + a bit of overhead")
    print(
        "DOK - stores everything in a dictionary, using indices as keys and values as values. Flexible but usually extremely high overhead, due to both Python's dict structure and the tuple keys")

    num_rows = temp_mtx.shape[0]
    num_cols = temp_mtx.shape[1]
    print("\nMemory usage:")
    results = [["Format", "Theoretical Size (bytes)", "Actual Size (bytes)", "Overhead Ratio (percent)", "Overhead to Base (percent)"],
               ["Base", str(base_bytes), str(base_bytes), "0", "0"]]
    for fmt in format_options:
        mtx = load_mm_file(args.input, fmt, False)
        if mtx is None:
            parser.error("input file format should be MatrixMarket, with .mtx extension")

        new_result = [fmt.upper()]
        theoretical_size = -1
        if fmt == "coo":
            theoretical_size = nnz * 4 * 2 + nnz * 8
        elif fmt == "csr":
            theoretical_size = (num_rows + 1) * 4 + nnz * 4 + nnz * 8
        elif fmt == "csc":
            theoretical_size = (num_cols + 1) * 4 + nnz * 4 + nnz * 8
        elif fmt == "dia":  # Calculation assumes naive layout without optimizations. Real result is optimized, which results in this being larger than final result
            num_diagonals = mtx.data.shape[0]
            theoretical_size = num_diagonals * 4 + num_diagonals * num_rows * 8
        elif fmt == "bsr":
            num_blocks = mtx.indices.size
            block_size = mtx.blocksize
            theoretical_size = num_blocks * (block_size[0] * block_size[1]) * 8 + num_blocks * 4 + (
                    int(num_rows / block_size[0]) + 1) * 4
        elif fmt == "lil":
            theoretical_size = nnz * 4 + nnz * 8 + num_rows * 4
        elif fmt == "dok":
            theoretical_size = nnz * 2 * 4 + nnz * 8  # This excludes dict structure overhead
        new_result.append(str(theoretical_size))

        actual_size = asizeof.asizeof(mtx)
        new_result.append(str(actual_size))

        overhead_ratio = ((actual_size - theoretical_size) / theoretical_size) * 100
        new_result.append(f"{overhead_ratio:.2f}")

        overhead_to_base = ((actual_size - base_bytes) / base_bytes) * 100
        new_result.append(f"{overhead_to_base:.2f}")

        results.append(new_result)

    col_widths = [max(len(item) for item in col) for col in zip(*results)]
    for res in results:
        print("    ".join(f"{item.ljust(width)}" for item, width in zip(res, col_widths)))

    if args.output is not None:
        arr = np.array(results)
        np.savetxt(args.output, arr, fmt='%s', delimiter=', ')
except Exception as e:
    print(e)




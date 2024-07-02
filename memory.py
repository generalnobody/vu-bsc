# Measure memory usage of sparse matrices and compare to theoretical requirements

from pympler import asizeof
import json
import argparse
import numpy as np
from loader import *

# Check if the script has been imported as a module
if __name__ != "__main__":
    raise ImportError("This script cannot be imported as a module")

try:
    # Load formats dictionary form JSON
    with open("./dicts.json", "r") as read_file:
        dicts = json.load(read_file)
        format_options = list(dicts['formats_dict'].keys())[:-1]

    # Define arguments
    parser = argparse.ArgumentParser(description="calculates the number of bytes used by each format with the provided matrix (does not support pytorch library)")

    parser.add_argument("-i", "--input", help="input file, MatrixMarket format", required=True)
    parser.add_argument('-o', '--output', help="CSV file to output result to; if not specified, only prints result to stdout (optional)")

    args = parser.parse_args()

    if args.output is not None and not args.output.endswith(".csv"):
        parser.error("input file format should be MatrixMarket, with .mtx extension")

    # Load matrix and get its basic statistics
    temp_mtx = load_mm_file(args.input, 'coo', False)
    if temp_mtx is None:
        parser.error("unknown input file format")
    nnz = temp_mtx.nnz
    entry_type = temp_mtx.dtype
    entry_size = entry_type.itemsize
    base_bytes = nnz * entry_size
    print(f"Number of non-zero entries in matrix is {nnz}. The type is {entry_type} with size {entry_size} bytes.\n"
          f"The non-zero entries require {base_bytes} bytes.")

    # Prepare results table
    num_rows = temp_mtx.shape[0]
    num_cols = temp_mtx.shape[1]
    print("\nMemory usage:")
    results = [["Format", "Theoretical Size (bytes)", "Actual Size (bytes)", "Overhead Ratio (percent)", "Overhead to Base (percent)"],
               ["Base", str(base_bytes), str(base_bytes), "0", "0"]]
    for fmt in format_options:
        # Load matrix in different formats
        mtx = load_mm_file(args.input, fmt, False)
        if mtx is None:
            parser.error("input file format should be MatrixMarket, with .mtx extension")

        # Calculate theoretically required amount of memory for the matrix in a particular format
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
            theoretical_size = nnz * 4 + nnz * 8
        elif fmt == "dok":
            theoretical_size = nnz * 2 * 4 + nnz * 8  # This excludes dict structure overhead
        new_result.append(str(theoretical_size))

        # Measure loaded matrix size
        actual_size = asizeof.asizeof(mtx)
        new_result.append(str(actual_size))

        overhead_ratio = ((actual_size - theoretical_size) / theoretical_size) * 100
        new_result.append(f"{overhead_ratio:.2f}")

        overhead_to_base = ((actual_size - base_bytes) / base_bytes) * 100
        new_result.append(f"{overhead_to_base:.2f}")

        # Populate results table
        results.append(new_result)

    # Style and print table to stdout
    col_widths = [max(len(item) for item in col) for col in zip(*results)]
    for res in results:
        print("    ".join(f"{item.ljust(width)}" for item, width in zip(res, col_widths)))

    # If location provided, output table to file (CSV)
    if args.output is not None:
        arr = np.array(results)
        np.savetxt(args.output, arr, fmt='%s', delimiter=', ')
except Exception as e:
    print(e)
    exit(1)




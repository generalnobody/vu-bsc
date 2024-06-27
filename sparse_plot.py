# Script responsible for plotting sparsity patterns of (multiple) Sparse Matrices

import os
import argparse
from math import ceil

from loader import load_mm_file
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Check if the script has been imported as a module
if __name__ != "__main__":
    raise ImportError("This script cannot be imported as a module")

try:
    # Define arguments
    parser = argparse.ArgumentParser(
        description="plots ")
    parser.add_argument("-f", "--file", help="path to MatrixMarket file(s) (multiple possible)", nargs='+', required=True)
    parser.add_argument("-o", "--output", help="specifies output file", required=True)
    args = parser.parse_args()

    # Sort files alphabetically
    args.file.sort(key=lambda s: s.lower())

    num_args = len(args.file)
    num_rows = int(ceil(num_args / 2))

    matrices = []

    # Load all matrices
    for file in args.file:
        matrices.append(load_mm_file(file, 'coo', False))

    # Define subplots
    fig, axes = plt.subplots(num_rows, 2)

    if num_rows > 1:
        axs = axes.flatten()
    else:
        axs = [axes]

    # For all matrices, plot their sparsity patterns
    for i, matrix in enumerate(matrices):
        axs[i].spy(matrix, color='black', markersize=.3)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        title = os.path.basename(args.file[i]).split('.')[0]
        axs[i].set_title(f"{chr(i + 97)}) {title}")

    # Clear any leftover plots
    for j in range(len(matrices), len(axs)):
        fig.delaxes(axs[j])

    # Save plots to file
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
except Exception as e:
    print(e)
    exit(1)

import os
import argparse
from math import ceil
from scipy import ndimage

from loader import load_mm_file
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

parser = argparse.ArgumentParser(
    description="plots ")

parser.add_argument("-f", "--file", help="path to MatrixMarket file(s) (multiple possible)", nargs='+', required=True)
parser.add_argument("-o", "--output", help="specifies output file", required=True)

args = parser.parse_args()

try:
    args.file.sort(key=lambda s: s.lower())

    num_args = len(args.file)
    num_rows = int(ceil(num_args / 2))

    matrices = []

    for file in args.file:
        matrices.append(load_mm_file(file, 'coo', False))

    fig, axes = plt.subplots(num_rows, 2)

    if num_rows > 1:
        axs = axes.flatten()
    else:
        axs = [axes]

    for i, matrix in enumerate(matrices):
        axs[i].spy(matrix, color='black', markersize=.3)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        title = os.path.basename(args.file[i]).split('.')[0]
        axs[i].set_title(f"{chr(i + 97)}) {title}")

    for j in range(len(matrices), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()


except Exception as e:
    print(e)

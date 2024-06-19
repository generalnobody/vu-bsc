# Script responsible for taking benchmarking results and plotting them in clear plots
# Run separately from the main script because results used for the final thesis are generated on remote DAS-5 cluster
import os
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import statistics as st
import numpy as np
from math import ceil

matplotlib.use('Agg')


def format_float(value):
    try:
        return "{:.2e}".format(float(value))
    except ValueError:
        return value


# This function plots the results in a boxplot. If there are pytorch results, includes those in the result.
# It plots the results per operation, meaning that for each tested function, it shows the performance of each format and, if available, each format using pytorch too
def plot_results(data, pytorch_data, output, output_format, dicts):
    results_dict = {}
    for mode in list(dicts['modes_dict'].keys())[:-1]:
        results_dict[mode] = []
    for fmt in data['data']:
        for res in fmt['results']:
            times = [x * 1000 for x in res['time']]  # Gets the times in milliseconds (ms)
            results_dict[res['mode']].append({'format': fmt['format'].upper(), 'time': times})

    if pytorch_data is not None:
        for fmt in pytorch_data['data']:
            for res in fmt['results']:
                times = [x * 1000 for x in res['time']]  # Gets the times in milliseconds (ms)
                results_dict[res['mode']].append({'format': f"{fmt['format'].upper()}\n(PyTorch)", 'time': times})

    num_iters = len(results_dict)
    num_rows = int(ceil(num_iters / 2))

    fig, axes = plt.subplots(num_rows, 2, figsize=(16, num_rows * 5), sharex='all', sharey='all', subplot_kw={'xscale': 'log', 'xlabel': 'Time (ms)', 'ylabel': 'Formats'})

    if num_rows > 1:
        axs = axes.flatten()
    else:
        axs = [axes]

    for i, (mode, res) in enumerate(results_dict.items()):
        group_data = []
        group_labels = []

        for d in res:
            group_data.append(d['time'])
            group_labels.append(d['format'])

        title = dicts['modes_dict'][mode]
        savepath = f"{output}/{dicts['modes_dict'][mode]}.{output_format}"

        axs[i].boxplot(group_data, labels=group_labels, vert=0)
        axs[i].set_title(f"{chr(i + 97)}) {title}")

    for j in range(len(results_dict), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(f"{output}/plots.{output_format}")
    plt.close()


parser = argparse.ArgumentParser(
    description="shows the results of the sparse matrix benchmarking script in clear formats")

parser.add_argument("-f", "--file", help="path to JSON file generated with sparse matrix benchmarking", required=True)
parser.add_argument("-ptf", "--pytorch_file", help="path to JSON file generated with pytorch benchmarking")
parser.add_argument("-o", "--output",
                    help="specifies the folder in which to save the generated plot(s) (default: ./plots)",
                    default="./plots")
parser.add_argument("-fmt", "--format", help="specifies the output files format (default: pdf)", default="pdf")

args = parser.parse_args()

try:
    with open(args.file, "r") as read_file:
        data = json.load(read_file)

    with open("./dicts.json", "r") as read_file:
        dicts = json.load(read_file)

    cleaned_path = args.output.rstrip('/')
    if not os.path.exists(cleaned_path):
        os.makedirs(cleaned_path)

    pytorch_data = None
    if args.pytorch_file is not None:
        with open(args.pytorch_file, "r") as read_file:
            pytorch_data = json.load(read_file)

    plot_results(data, pytorch_data, cleaned_path, args.format, dicts)
    stats = [
        ["Format", "Benchmark", "Min", "P25", "P50 (Median)", "P75", "Max", "Standard Deviation", "Mean", "Variance",
         "Range"]]
    # For-loops are repeated, because lengths of arrays in data and pytorch_data can be different
    for fmt in data['data']:
        for res in fmt['results']:
            percentiles = st.quantiles(res['time'], n=4)
            stats.append([
                fmt['format'].upper(),
                dicts['modes_dict'][res['mode']],
                min(res['time']) * 1000,
                percentiles[0] * 1000,
                st.median(res['time']) * 1000,
                percentiles[2] * 1000,
                max(res['time']) * 1000,
                st.stdev(res['time']) * 1000,
                st.mean(res['time']) * 1000,
                st.variance(res['time']) * 1000,
                max(res['time']) - min(res['time']) * 1000
            ])
    if pytorch_data is not None:
        for fmt in pytorch_data['data']:
            for res in fmt['results']:
                percentiles = st.quantiles(res['time'], n=4)
                stats.append([
                    f"{fmt['format'].upper()} - PyTorch",
                    dicts['modes_dict'][res['mode']],
                    min(res['time']) * 1000,
                    percentiles[0] * 1000,
                    st.median(res['time']) * 1000,
                    percentiles[2] * 1000,
                    max(res['time']) * 1000,
                    st.stdev(res['time']) * 1000,
                    st.mean(res['time']) * 1000,
                    st.variance(res['time']) * 1000,
                    max(res['time']) - min(res['time']) * 1000
                ])
    vectorized_format = np.vectorize(format_float)
    arr = np.array(stats)
    arr[1:, 2:] = vectorized_format(arr[1:, 2:])
    np.savetxt(f"{cleaned_path}/stats.csv", arr, fmt='%s', delimiter=', ')
except Exception as e:
    print(e)

# Script responsible for taking benchmarking results and plotting them in clear plots
# Run separately from the main script because results used for the final thesis are generated on remote DAS-5 cluster
import os
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import statistics as st
import numpy as np

matplotlib.use('Agg')


def plot_boxplot(data, labels, title, ylabel, save_path, show):
    plt.figure(figsize=(12, 8))
    plt.boxplot(data, labels=labels, vert=0)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    if show is True:
        plt.show()
    else:
        plt.close()


# This function plots the results for a particular chosen mode. Includes as many formats as available
def plot_mode_benchmark(data, output, show, dicts):
    results_dict = {}
    for mode in list(dicts['modes_dict'].keys())[:-1]:
        results_dict[mode] = []
    for fmt in data['data']:
        for res in fmt['results']:
            times = [x * 1000 for x in res['time']]  # Gets the times in milliseconds (ms)
            results_dict[res['mode']].append({'format': fmt['format'], 'time': times})

    for mode, res in results_dict.items():
        group_data = []
        group_labels = []

        for d in res:
            group_data.append(d['time'])
            group_labels.append(d['format'].upper())

        title = dicts['modes_dict'][mode]
        ylabel = "Formats"
        savepath = f"{output}/mode/{dicts['modes_dict'][mode]}.png"
        plot_boxplot(group_data, group_labels, title, ylabel, savepath, show)


# This function plots the results for a particular chosen format. Includes as many modes as available
def plot_format_benchmark(data, output, show, dicts):
    for fmt in data['data']:
        plt.figure(figsize=(12, 8))
        group_data = []
        group_labels = []
        for res in fmt['results']:
            times = [x * 1000 for x in res['time']]  # Gets the times in milliseconds (ms)
            group_data.append(times)
            group_labels.append(f"{dicts['modes_dict'][res['mode']]}")

        title = f"{dicts['formats_dict'][fmt['format']]} ({fmt['format'].upper()})"
        ylabel = "Modes"
        savepath = f"{output}/format/{dicts['formats_dict'][fmt['format']]}.png"
        plot_boxplot(group_data, group_labels, title, ylabel, savepath, show)


parser = argparse.ArgumentParser(
    description="shows the results of the sparse matrix benchmarking script in clear formats")

parser.add_argument("-f", "--file", help="path to JSON file generated with sparse matrix benchmarking", required=True)
parser.add_argument("--plot", help="whether to plot based on the function, based on the format or both",
                    choices={"mode", "format", "both"}, required=True)
parser.add_argument("-o", "--output",
                    help="specifies the folder in which to save the generated plot(s) (default: ./plots)",
                    default="./plots")
parser.add_argument("-s", "--show",
                    help="whether to show the generated plot(s) (default: False) (not recommended when provided JSON file contains multiple results)",
                    action="store_true")

args = parser.parse_args()

try:
    with open(args.file, "r") as read_file:
        data = json.load(read_file)

    with open("./dicts.json", "r") as read_file:
        dicts = json.load(read_file)

    cleaned_path = args.output.rstrip('/')
    mode_path = f"{cleaned_path}/mode"
    if not os.path.exists(mode_path):
        os.makedirs(mode_path)
    format_path = f"{cleaned_path}/format"
    if not os.path.exists(format_path):
        os.makedirs(format_path)

    if args.plot == "mode" or args.plot == "both":
        plot_mode_benchmark(data, cleaned_path, args.show, dicts)
    if args.plot == "format" or args.plot == "both":
        plot_format_benchmark(data, cleaned_path, args.show, dicts)

    # TODO: write function that prints the statistics for all the results in the benchmark\
    stats = [
        ["Format", "Benchmark", "Min", "25%", "50% (Median)", "75%", "Max", "Standard Deviation", "Mean", "Variance",
         "Range"]]
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
    arr = np.array(stats)
    np.savetxt(f"{cleaned_path}/stats.csv", arr, fmt='%s', delimiter=', ')
except Exception as e:
    print(e)

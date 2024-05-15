# Script responsible for taking benchmarking results and plotting them in clear plots

import matplotlib.pyplot as plt
import statistics
import numpy as np

def plot_results(results, reps):
    plt.figure(figsize=(7,5))
    plt.bar(results)
    plt.title("Results over %s measurements" % reps)
    plt.show()
    print("Average of these results is: %s" % statistics.mean(results))

def plot_full_results(modes, results, reps):
    for i in range(len(modes)):
        plot_results(results[i], reps)


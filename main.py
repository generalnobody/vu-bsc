# Main script that handles everything surrounding the benchmarking functionality

import argparse
import sys
import json
import warnings
import numpy as np

from loader import load_mm_file
from functions import *
from benchmark import *


# Add custom help window for Sparse Matrix formats to argparse
class FormatHelpAction(argparse.Action):
    def __call__(self, prs, namespace, values, option_string=None):
        print("usage: %s [...] --format %s [...]" % (sys.argv[0], "{" + ",".join(format_options) + "}"))
        print("\noptions:")
        for key, value in formats_dict.items():
            print(f"  {key}: {value}")
        prs.exit()


# Add custom help window for functions to execute to argparse
class ModeHelpAction(argparse.Action):
    def __call__(self, prs, namespace, values, option_string=None):
        print("usage: %s [...] --mode %s [...]" % (sys.argv[0], "{" + ",".join(mode_options) + "}"))
        print("\noptions:")
        for key, value in modes_dict.items():
            print(f"  {key}: {value}")
        prs.exit()


# Call benchmark function, providing it with the function to execute and its arguments
def perform_benchmark(mode, mtx_a, mtx_b=None, idx=-1, scl=-1, reps=0):
    # Prepare results dictionary
    benchmark_results = {'mode': mode}

    # Depending on the mode, call a different function, populate results dictionary
    if mode == "add":
        benchmark_results['time'] = benchmark(mtx_addition, mtx_a, mtx_b, reps=reps)
    elif mode == "sub":
        benchmark_results['time'] = benchmark(mtx_subtraction, mtx_a, mtx_b, reps=reps)
    elif mode == "sm":
        benchmark_results['time'] = benchmark(mtx_scalar_multiplication, scl, mtx_a, reps=reps)
    elif mode == "mvm":
        vec = None
        if mtx_a.__module__.startswith('torch'):
            dense_mtx = mtx_a.to_dense()
            vec = dense_mtx[idx]
        elif mtx_a.__module__.startswith('scipy'):
            vec = mtx_a.getrow(idx)
        benchmark_results['time'] = benchmark(mtx_matrix_vector_multiplication, mtx_a, vec, reps=reps)
    elif mode == "mmm":
        benchmark_results['time'] = benchmark(mtx_matrix_matrix_multiplication, mtx_a, mtx_b, reps=reps)
    elif mode == "tps":
        benchmark_results['time'] = benchmark(mtx_transposition, mtx_a, reps=reps)

    return benchmark_results


# Handle benchmark executing at the format level, meaning that, with a set format, handle benchmarking related to modes
def run_format(args):
    matrix_b = None
    row_index = 0
    # Load primary matrix
    matrix_a = load_mm_file(args.path_a, args.format, args.pytorch)
    if matrix_a is None:
        parser.exit()

    # Load secondary matrix (in my results, this is the same as primary matrix)
    if args.mode == "add" or args.mode == "sub" or args.mode == "mmm" or args.mode == "full":
        if args.path_b is None:
            parser.error("option '%s' required for mode '%s'" % ("--path_b", args.mode))
        else:
            matrix_b = load_mm_file(args.path_b, args.format, args.pytorch)
            if matrix_b is None:
                parser.exit()
    # Ensure scalar value is defined if needed
    if (args.mode == "sm" or args.mode == "full") and args.scalar is None:
        parser.error("option '%s' required for mode '%s'" % ("--scalar", args.mode))
    # Select vector for mvm based on user selection or otherwise randomly
    if args.mode == "mvm" or args.mode == "full":
        num_rows = matrix_a.shape[0]
        if args.index is None:
            row_index = np.random.randint(num_rows)
        else:
            row_index = args.index

    # Define results dictionary
    fmt_results = {'format': args.format, 'results': []}

    # Execute parameter format's functions based on arguments, populate results dictionary
    if args.mode == "full":
        for mode in mode_options[:-1]:
            fmt_results['results'].append(
                perform_benchmark(mode, matrix_a, mtx_b=matrix_b, idx=row_index, scl=args.scalar, reps=args.benchmark)
            )
    else:
        fmt_results['results'].append(
            perform_benchmark(args.mode, matrix_a, mtx_b=matrix_b, idx=row_index, scl=args.scalar,
                              reps=args.benchmark)
        )

    return fmt_results


#####################################################################################################################
# Main part of this script

# Check if the script has been imported as a module
if __name__ != "__main__":
    raise ImportError("This script cannot be imported as a module")

try:
    # Load formats and modes dict from dicts.json file
    with open("./dicts.json", "r") as read_file:
        dicts = json.load(read_file)

    formats_dict = dicts['formats_dict']
    modes_dict = dicts['modes_dict']

    format_options = list(formats_dict.keys())
    mode_options = list(modes_dict.keys())

    # Parse arguments
    parser = argparse.ArgumentParser(description="sparse matrix benchmarking script")

    help_group = parser.add_argument_group('additional help')
    help_group.add_argument('--format_help', help="show additional information about the possible formats and exit",
                            action=FormatHelpAction, nargs=0)
    help_group.add_argument('--mode_help', help="show additional information about the possible modes and exit",
                            action=ModeHelpAction, nargs=0)

    parser.add_argument('-b', '--benchmark', type=int, required=True,
                        help="select the number of times to benchmark the chosen mode(s) (minimum 1)")
    parser.add_argument('--format', choices=format_options, help="choose sparse matrix format(s) to use (required)",
                        required=True)
    parser.add_argument('--mode', choices=mode_options, help="choose the function(s) to benchmark (required)",
                        required=True)
    parser.add_argument('--path_a', help="path to the main matrix to be used for the benchmark (required)",
                        required=True)
    parser.add_argument('--path_b',
                        help="path to the secondary matrix to be used for the benchmark (required for modes add, sub and mmm)")
    parser.add_argument('--scalar', type=int, help="scalar value used for the benchmark (required for mode sm)")
    parser.add_argument('--index', type=int,
                        help="index of the row in the matrix to select as vector (optional for mode mvm; if not chosen, selected randomly)")
    parser.add_argument('-o', '--out', help="path to save the result to, otherwise it gets printed to stdout (JSON format)")
    parser.add_argument('-pt', '--pytorch', action="store_true",
                        help="use pytorch instead of scipy (only works with coo, csr, csc and bsr formats)")

    parser_args = parser.parse_args()

    if parser_args.benchmark < 1:
        parser.error("value for --benchmark must at least 1")

    if parser_args.out is not None and not parser_args.out.endswith(".json"):
        parser.error("output file should be in .json format")

    if parser_args.pytorch:
        warnings.filterwarnings("ignore", category=UserWarning)

    # Prepare results dictionary
    results = {'data': []}

    # Execute benchmarks based on parsed arguments and add results to results dictionary
    if parser_args.format == "all":
        for fmt in format_options[:-1]:
            if parser_args.pytorch and fmt not in ['coo', 'csr', 'csc', 'bsr']:
                continue
            parser_args.format = fmt
            results['data'].append(run_format(parser_args))
    else:
        if parser_args.pytorch and parser_args.format not in ['coo', 'csr', 'csc', 'bsr']:
            parser.error("format '{}' is not supported by pytorch".format(parser_args))
        results['data'].append(run_format(parser_args))

    # Output results as JSON to stdout or defined file
    if parser_args.out is None:
        print(json.dumps(results, indent=4))
    else:
        with open(parser_args.out, "w") as write_file:
            json.dump(results, write_file, indent=4)
except Exception as e:
    print(e)
    exit(1)


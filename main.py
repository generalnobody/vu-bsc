# Main script that handles everything surrounding the benchmarking functionality

import argparse
import sys
import json

from loader import load_mm_file
from functions import *
from benchmark import *


class FormatHelpAction(argparse.Action):
    def __call__(self, prs, namespace, values, option_string=None):
        print("usage: %s [...] --format %s [...]" % (sys.argv[0], "{" + ",".join(format_options) + "}"))
        print("\noptions:")
        for key, value in formats_dict.items():
            print(f"  {key}: {value}")
        prs.exit()


class ModeHelpAction(argparse.Action):
    def __call__(self, prs, namespace, values, option_string=None):
        print("usage: %s [...] --mode %s [...]" % (sys.argv[0], "{" + ",".join(mode_options) + "}"))
        print("\noptions:")
        for key, value in modes_dict.items():
            print(f"  {key}: {value}")
        prs.exit()


def perform_benchmark(mode, mtx_a, mtx_b=None, idx=-1, scl=-1, reps=0):
    benchmark_results = {'mode': mode}
    if mode == "spr":
        benchmark_results['time'] = benchmark(mtx_splice_row, mtx_a, idx, reps=reps)
    elif mode == "spc":
        benchmark_results['time'] = benchmark(mtx_splice_column, mtx_a, idx, reps=reps)
    elif mode == "add":
        benchmark_results['time'] = benchmark(mtx_addition, mtx_a, mtx_b, reps=reps)
    elif mode == "sub":
        benchmark_results['time'] = benchmark(mtx_subtraction, mtx_a, mtx_b, reps=reps)
    elif mode == "sm":
        benchmark_results['time'] = benchmark(mtx_scalar_multiplication, scl, mtx_a, reps=reps)
    elif mode == "mvm":
        spliced_vector = mtx_splice_row(mtx_a, idx)
        benchmark_results['time'] = benchmark(mtx_matrix_vector_multiplication, mtx_a, spliced_vector, reps=reps)
    elif mode == "mmm":
        benchmark_results['time'] = benchmark(mtx_matrix_matrix_multiplication, mtx_a, mtx_b, reps=reps)
    elif mode == "tps":
        benchmark_results['time'] = benchmark(mtx_transposition, mtx_a, reps=reps)
    # elif mode == "inv":  # Removed for now, since it only supports CSR and CSC formats
    #     benchmark_results['time'] = benchmark(mtx_inversion, mtx_a, reps=reps)

    return benchmark_results


def run_format(args):
    matrix_b = None
    column_index =0
    row_index = 0
    matrix_a = load_mm_file(args.path_a, args.format)
    if matrix_a is None:
        parser.exit()  # TODO: catch exceptions here and everywhere else

    if args.mode == "add" or args.mode == "sub" or args.mode == "mmm" or args.mode == "full":
        if args.path_b is None:
            parser.error("option '%s' required for mode '%s'" % ("--path_b", args.mode))
        else:
            matrix_b = load_mm_file(args.path_b, args.format)
            if matrix_b is None:
                parser.exit()
    if (args.mode == "sm" or args.mode == "full") and args.scalar is None:
        parser.error("option '%s' required for mode '%s'" % ("--scalar", args.mode))
    if args.mode == "spc":
        num_columns = matrix_a.shape[1]
        if args.index is None:
            column_index = np.random.randint(num_columns)
        else:
            column_index = args.index
    if args.mode == "spr" or args.mode == "mvm" or args.mode == "full":
        num_rows = matrix_a.shape[0]
        if args.index is None:
            row_index = np.random.randint(num_rows)
        else:
            row_index = args.index

    fmt_results = {'format': args.format, 'results': []}

    if args.mode == "full":
        for mode in mode_options[:-1]:
            idx = -1
            if mode == "spc":
                idx = column_index
            elif mode == "spr" or args.mode == "mvm":
                idx = row_index
            fmt_results['results'].append(
                perform_benchmark(mode, matrix_a, mtx_b=matrix_b, idx=idx, scl=args.scalar, reps=args.benchmark)
            )
    else:
        idx = -1
        if args.mode == "spc":
            idx = column_index
        elif args.mode == "spr" or args.mode == "mvm":
            idx = row_index
        fmt_results['results'].append(
            perform_benchmark(args.mode, matrix_a, mtx_b=matrix_b, idx=idx, scl=args.scalar, reps=args.benchmark)
        )

    return fmt_results


# Removed inversion, check later if it is actually possible
with open("./dicts.json", "r") as read_file:
    dicts = json.load(read_file)

formats_dict = dicts['formats_dict']
modes_dict = dicts['modes_dict']

format_options = list(formats_dict.keys())
mode_options = list(modes_dict.keys())

parser = argparse.ArgumentParser(description="sparse matrix benchmarking script")

help_group = parser.add_argument_group('additional help')
help_group.add_argument('--format_help', help="show additional information about the possible formats and exit",
                        action=FormatHelpAction, nargs=0)
help_group.add_argument('--mode_help', help="show additional information about the possible modes and exit",
                        action=ModeHelpAction, nargs=0)

parser.add_argument('-b', '--benchmark', type=int, required=True,
                    help="select the number of times to benchmark the chosen mode (minimum 1)")
parser.add_argument('--format', choices=format_options, help="choose sparse matrix format to use (Required)",
                    required=True)
parser.add_argument('--mode', choices=mode_options, help="choose the function to execute (Required)", required=True)
parser.add_argument('--path_a', help="path to the main matrix to be used for the chosen function (Required)",
                    required=True)
parser.add_argument('--path_b',
                    help="path to the secondary matrix to be used for the chosen function (Required for modes add, sub and mmm)")
parser.add_argument('--scalar', type=int, help="scalar value used for the chosen function (Required for mode sm)")
parser.add_argument('--index', type=int,
                    help="index of the row/column in the matrix to select for splicing or as vector (Optional for modes spr, spc and mvm)")
parser.add_argument('-o', '--out', help="path to save the result to, otherwise it gets printed to stdout (JSON format)")
# parser.add_argument('-t', '--threads', help="number of threads to use when running the code (default = 1) (currently not implemented)")

parser_args = parser.parse_args()

if parser_args.benchmark < 1:
    parser.error("value for --benchmark must at least 1")

results = {'data': []}

if parser_args.format == "all":
    for fmt in format_options[:-1]:
        parser_args.format = fmt
        results['data'].append(run_format(parser_args))
else:
    results['data'].append(run_format(parser_args))

if parser_args.out is None:
    print(json.dumps(results, indent=4))
else:
    with open(parser_args.out, "w") as write_file:
        json.dump(results, write_file, indent=4)

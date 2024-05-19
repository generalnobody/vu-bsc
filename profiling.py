import argparse
import sys
import json
import cProfile

from loader import load_mm_file
from functions import *

with open("./dicts.json", "r") as read_file:
    dicts = json.load(read_file)

formats_dict = dicts['formats_dict']
modes_dict = dicts['modes_dict']

format_options = list(formats_dict.keys())
mode_options = list(modes_dict.keys())

parser = argparse.ArgumentParser(description="sparse matrix function testing script")


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


help_group = parser.add_argument_group('additional help')
help_group.add_argument('--format_help', help="show additional information about the possible formats and exit",
                        action=FormatHelpAction, nargs=0)
help_group.add_argument('--mode_help', help="show additional information about the possible modes and exit",
                        action=ModeHelpAction, nargs=0)

parser.add_argument('--format', choices=format_options, help="choose sparse matrix format to use (Required)",
                    required=True)
parser.add_argument('--mode', choices=mode_options, help="choose the function to execute (Required)", required=True)
parser.add_argument('--path_a', help="path to the main matrix to be used for the chosen function (Required)",
                    required=True)
parser.add_argument('--path_b',
                    help="path to the secondary matrix to be used for the chosen function (Required for modes add, sub and mmm)")
parser.add_argument('--scalar', type=int, help="scalar value used for the chosen function (Required for mode sm)")
parser.add_argument('--index', type=int,
                    help="index of the row/column in the matrix to select for splicing or as vector (Optional for mode mvm)")

args = parser.parse_args()

matrix_b = None
row_index = -1
matrix_a = load_mm_file(args.path_a, args.format)
if matrix_a is None:
    parser.exit()  # TODO: catch exceptions here and everywhere else

if args.mode == "add" or args.mode == "sub" or args.mode == "mmm":
    if args.path_b is None:
        parser.error("option '%s' required for mode '%s'" % ("--path_b", args.mode))
    else:
        matrix_b = load_mm_file(args.path_b, args.format)
        if matrix_b is None:
            parser.exit()
if args.mode == "sm" and args.scalar is None:
    parser.error("option '%s' required for mode '%s'" % ("--scalar", args.mode))
if args.mode == "mvm":
    num_rows = matrix_a.shape[0]
    if args.index is None:
        row_index = np.random.randint(num_rows)
    else:
        row_index = args.index

profiler = cProfile.Profile()

result = None
if args.mode == "add":
    profiler.enable()
    result = mtx_addition(matrix_a, matrix_b)
    profiler.disable()
elif args.mode == "sub":
    profiler.enable()
    result = mtx_subtraction(matrix_a, matrix_b)
    profiler.disable()
elif args.mode == "sm":
    profiler.enable()
    result = mtx_scalar_multiplication(args.scalar, matrix_a)
    profiler.disable()
elif args.mode == "mvm":  # TODO: Test whether different size matrices/vectors work
    sp_vector = matrix_a.getrow(row_index)
    profiler.enable()
    result = mtx_matrix_vector_multiplication(matrix_a, sp_vector)
    profiler.disable()
elif args.mode == "mmm":  # TODO: Test whether different size matrices/vectors work
    profiler.enable()
    result = mtx_matrix_matrix_multiplication(matrix_a, matrix_b)
    profiler.disable()
elif args.mode == "tps":
    profiler.enable()
    result = mtx_transposition(matrix_a)
    profiler.disable()

if result is not None:
    print(result)

profiler.print_stats()
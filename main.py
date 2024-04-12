# Entry point of the project. Responsible for loading matrix files and combining all other functions into a single workflow

import argparse
import sys

from loader import load_mm_file
from functions import *
from results import *
from benchmark import *

global matrix_b, sp_vector, row_index, column_index

formats_dict = {
    'coo': "Coordinate List",
    'csr': "Compressed Sparse Row",
    'csc': "Compressed Sparse Column",
    'dia': "Diagonal Storage",
    'bsr': "Block Compressed Row Storage",
    'lil': "List of Lists",
    'dok': "Dictionary of Keys"
}

modes_dict = {
    'spr': "Row splicing",
    'spc': "Column splicing",
    'add': "Matrix addition",
    'sub': "Matrix subtraction",
    'sm': " Scalar multiplication",
    'mvm': "Sparse matrix-vector multiplication",
    'mmm': "Sparse matrix-matrix multiplication",
    "tps": "Transposition",
    "inv": "Inversion (if possible)",
    "full": "Run all above-mentioned functions"
}

format_options = list(formats_dict.keys())
mode_options = list(modes_dict.keys())

parser = argparse.ArgumentParser(description="sparse matrix benchmarking script")


class FormatHelpAction(argparse.Action):
    def __call__(self, prs, namespace, values, option_string=None):
        # print("Possible formats used for testing:")
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


help_group = parser.add_argument_group('Additional help')
help_group.add_argument('--format_help', help="show additional information about the possible formats and exit",
                        action=FormatHelpAction, nargs=0)
help_group.add_argument('--mode_help', help="show additional information about the possible modes and exit",
                        action=ModeHelpAction, nargs=0)

parser.add_argument('-b', '--benchmark', action='store_true', help="select if you want to run benchmark")
parser.add_argument('--format', choices=format_options, help="choose sparse matrix format to use (Required)",
                    required=True)
parser.add_argument('--mode', choices=mode_options, help="choose the function to execute (Required)", required=True)
parser.add_argument('--path_a', help="path to the main matrix to be used for the chosen function (Required)",
                    required=True)
parser.add_argument('--path_b',
                    help="path to the secondary matrix to be used for the chosen function (Required for modes add, sub and mmm)")
parser.add_argument('--scalar', help="scalar value used for the chosen function (Required for mode sm)")
parser.add_argument('--index',
                    help="index of the row/column in the matrix to select for splicing or as vector (Optional for modes spr, spc and mvm)",
                    type=int)

args = parser.parse_args()

matrix_a = load_mm_file(args.path_a, args.format)

if args.mode == "add" or args.mode == "sub" or args.mode == "mmm":
    if args.path_b is None:
        parser.error("option '%s' required for mode '%s'" % ("--path_b", args.mode))
    else:
        matrix_b = load_mm_file(args.path_b, args.format)
elif args.mode == "sm" and args.scalar is None:
    parser.error("option '%s' required for mode '%s'" % ("--scalar", args.mode))
elif args.mode == "spc":
    num_columns = matrix_a.shape[1]
    if args.index is None:
        column_index = np.random.randint(num_columns)
    else:
        column_index = args.index
elif args.mode == "spr" or args.mode == "mvm":
    num_rows = matrix_a.shape[0]
    if args.index is None:
        row_index = np.random.randint(num_rows)
    else:
        row_index = args.index
elif args.mode == "full":
    if args.path_b is None:
        parser.error("option '%s' required for mode '%s'" % ("--path_b", args.mode))
    elif args.scalar is None:
        parser.error("option '%s' required for mode '%s'" % ("--scalar", args.mode))

    num_rows = matrix_a.shape[0]
    if args.index is None:
        row_index = np.random.randint(num_rows)
    else:
        row_index = args.index

result = 0
if args.mode == "spr":
    result = mtx_splice_row(matrix_a, row_index)
elif args.mode == "spc":
    result = mtx_splice_column(matrix_a, column_index)
elif args.mode == "add":
    result = mtx_addition(matrix_a, matrix_b)
elif args.mode == "sub":
    result = mtx_subtraction(matrix_a, matrix_b)
elif args.mode == "sm":
    result = mtx_scalar_multiplication(args.scalar, matrix_a)
elif args.mode == "mvm":
    sp_vector = mtx_splice_row(matrix_a, row_index)
    result = mtx_matrix_vector_multiplication(matrix_a, sp_vector)
elif args.mode == "mmm":
    result = mtx_matrix_matrix_multiplication(matrix_a, matrix_b)
elif args.mode == "tps":
    result = mtx_transposition(matrix_a)
elif args.mode == "inv":
    result = mtx_inversion(matrix_a)

print(result)

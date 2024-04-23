# Entry point of the project. Responsible for loading matrix files and combining all other functions into a single workflow

import argparse
import sys

from loader import load_mm_file
from functions import *
from results import *
from benchmark import *
from output import *

global matrix_b, sp_vector, row_index, column_index

formats_dict = {
    'coo': "Coordinate List",
    'csr': "Compressed Sparse Row",
    'csc': "Compressed Sparse Column",
    'dia': "Diagonal Storage",
    'bsr': "Block Compressed Row Storage",
    'lil': "List of Lists",
    'dok': "Dictionary of Keys",
    'all': "All formats mentioned above"
}

modes_dict = {
    'spr': "Row splicing",
    'spc': "Column splicing",
    'add': "Matrix addition",
    'sub': "Matrix subtraction",
    'sm': " Scalar multiplication",
    'mvm': "Sparse matrix-vector multiplication",
    'mmm': "Sparse matrix-matrix multiplication",
    'tps': "Transposition",
    'inv': "Inversion (if possible)",
    'full': "Run all above-mentioned functions"
}

format_options = list(formats_dict.keys())
mode_options = list(modes_dict.keys())

parser = argparse.ArgumentParser(description="sparse matrix benchmarking script")


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

parser.add_argument('-b', '--benchmark', type=int, default=0,
                    help="select the number of times to benchmark the chosen mode (default = 0, disables benchmark and runs the chosen function once)")
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
parser.add_argument('-o', '--out', help="path to save the result to, otherwise it gets printed to stdout")

args = parser.parse_args()

if args.benchmark < 0:
    parser.error("value for --benchmark must be greater than or equal to 0")
elif args.benchmark == 0 and args.format == "all":
    parser.error(
        "format '%s' only for benchmarking; to use, set --benchmark to greater than 0 or select different mode" % args.format)

# TODO: turn part below into a function so that with all it can be called manually

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


def perform_benchmark(mode, mtx_a, mtx_b=None, idx=-1, scl=-1, reps=0):
    if mode == "spr":
        return benchmark(mtx_splice_row, mtx_a, idx, reps=reps)
    elif mode == "spc":
        return benchmark(mtx_splice_column, mtx_a, idx, reps=reps)
    elif mode == "add":
        return benchmark(mtx_addition, mtx_a, mtx_b, reps=reps)
    elif mode == "sub":
        return benchmark(mtx_subtraction, mtx_a, mtx_b, reps=reps)
    elif mode == "sm":
        return benchmark(mtx_scalar_multiplication, scl, mtx_a, reps=reps)
    elif mode == "mvm":
        sp_vector = mtx_splice_row(mtx_a, idx)
        return benchmark(mtx_matrix_vector_multiplication, mtx_a, sp_vector, reps=reps)
    elif mode == "mmm":
        return benchmark(mtx_matrix_matrix_multiplication, mtx_a, mtx_b, reps=reps)
    elif mode == "tps":
        return benchmark(mtx_transposition, mtx_a, reps=reps)
    elif mode == "inv":
        return benchmark(mtx_inversion, mtx_a, reps=reps)


if args.benchmark > 0:
    print("benchmarking...")
    result = []
    if args.mode == "full":
        for mode in mode_options[:-1]:
            idx = -1
            if mode == "spc":
                idx = column_index
            elif mode == "spr" or args.mode == "mvm":
                idx = row_index
            result.append(perform_benchmark(mode, matrix_a, mtx_b=matrix_b, idx=idx, scl=args.scalar, reps=args.benchmark))
    else:
        idx = -1
        if args.mode == "spc":
            idx = column_index
        elif args.mode == "spr" or args.mode == "mvm":
            idx = row_index
        result = perform_benchmark(args.mode, matrix_a, mtx_b=matrix_b, idx=idx, scl=args.scalar, reps=args.benchmark)
else:
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
    elif args.mode == "mvm":  # TODO: Test whether different size matrices/vectors work
        sp_vector = mtx_splice_row(matrix_a, row_index)
        result = mtx_matrix_vector_multiplication(matrix_a, sp_vector)
    elif args.mode == "mmm":  # TODO: Test whether different size matrices/vectors work
        result = mtx_matrix_matrix_multiplication(matrix_a, matrix_b)
    elif args.mode == "tps":
        result = mtx_transposition(matrix_a)
    elif args.mode == "inv":  # TODO: maybe remove this operation from tests, since it only supports CSC and CSR
        result = mtx_inversion(matrix_a)

    if args.out is None:
        print(result)
    else:
        write_mm_file(args.out, result)

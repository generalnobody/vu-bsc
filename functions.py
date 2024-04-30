# Script containing functions performing singular operations on provided sparse matrices. Time used for running the function is measured by the benchmark
from scipy.sparse.linalg import inv
import numpy as np
from memory_profiler import profile

@profile
def mtx_splice_row(mtx, row_index):
    return mtx.getrow(row_index)

@profile
def mtx_splice_column(mtx, column_index):
    return mtx.getcol(column_index)

@profile
def mtx_addition(mtx_a, mtx_b):
    return mtx_a + mtx_b

@profile
def mtx_subtraction(mtx_a, mtx_b):
    return mtx_a - mtx_b

@profile
def mtx_scalar_multiplication(scalar, mtx):
    return scalar * mtx

@profile
def mtx_matrix_vector_multiplication(mtx, vec):
    return mtx.multiply(vec)

@profile
def mtx_matrix_matrix_multiplication(mtx_a, mtx_b):
    return mtx_a.multiply(mtx_b)

@profile
def mtx_transposition(mtx):
    return mtx.transpose()


# def mtx_inversion(mtx):
#     try:
#         return inv(mtx)
#     except np.linalg.LinAlgError:
#         return "Matrix not invertible"

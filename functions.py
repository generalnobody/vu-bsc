# Script containing functions performing singular operations on provided sparse matrices. Time used for running the function is measured by the benchmark

def mtx_addition(mtx_a, mtx_b):
    return mtx_a + mtx_b


def mtx_subtraction(mtx_a, mtx_b):
    return mtx_a - mtx_b


def mtx_scalar_multiplication(scalar, mtx):
    return scalar * mtx


def mtx_matrix_vector_multiplication(mtx, vec):
    return mtx.multiply(vec)


def mtx_matrix_matrix_multiplication(mtx_a, mtx_b):
    return mtx_a.multiply(mtx_b)


def mtx_transposition(mtx):
    return mtx.transpose()

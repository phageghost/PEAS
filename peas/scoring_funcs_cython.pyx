# File: scoring_funcs_cython.pyx


import cython
import numpy

numpy.import_array()


cdef extern void c_compute_sum_table_2d(double*, size_t, size_t, size_t, double*)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_sum_table_2d(
        numpy.ndarray[double, ndim=2, mode='c'] data_matrix not None,
        long start_diagonal,
        long end_diagonal):
    """
    """
    matrix_size = data_matrix.shape[0]

    cdef numpy.ndarray[double, ndim=2, mode='c'] sum_table = numpy.zeros(shape=(matrix_size, matrix_size), dtype=float,
                                                                         order='C')

    c_compute_sum_table_2d(&data_matrix[0, 0], <size_t> matrix_size, <size_t> int(start_diagonal),
                           <size_t> int(end_diagonal), &sum_table[0, 0])

    return sum_table

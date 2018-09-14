# File: scoring_funcs_cython.pyx
import numpy
cimport numpy
import cython

numpy.import_array()

cdef extern void c_compute_sum_table_2d(double*, size_t, size_t, size_t, double*)
cdef extern void c_compute_sum_table_2d_shuffled(double*, size_t, size_t, size_t, double*)
cdef extern void c_compute_mean_table_2d_shuffled(double*, size_t, size_t, size_t, double*)

cdef extern void c_compute_sum_table_1d_shuffled(double*, size_t, size_t, double*)
cdef extern void c_compute_mean_table_1d_shuffled(double*, size_t, size_t, double*)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_sum_table_2d(
        numpy.ndarray[double, ndim=2, mode='c'] data_matrix not None,
        long start_diagonal,
        long end_diagonal):
    """
    """
    assert data_matrix.shape[0] == data_matrix.shape[1]
    matrix_size = data_matrix.shape[0]
    assert -matrix_size < start_diagonal < end_diagonal <= matrix_size

    cdef numpy.ndarray[double, ndim=2, mode='c'] sum_table = numpy.zeros(shape=(matrix_size, matrix_size), dtype=float,
                                                                         order='C')
    c_compute_sum_table_2d(&data_matrix[0, 0], <size_t> matrix_size, <size_t> int(start_diagonal),
                           <size_t> int(end_diagonal), &sum_table[0, 0])
    return sum_table

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_sum_table_2d_shuffled(
        numpy.ndarray[double, ndim=2, mode='c'] data_matrix not None,
        long start_diagonal,
        long end_diagonal):
    """
    """
    assert data_matrix.shape[0] == data_matrix.shape[1]
    matrix_size = data_matrix.shape[0]
    assert -matrix_size < start_diagonal < end_diagonal <= matrix_size

    cdef numpy.ndarray[double, ndim=2, mode='c'] sum_table = numpy.zeros(shape=(matrix_size, matrix_size), dtype=float,
                                                                         order='C')
    c_compute_sum_table_2d_shuffled(&data_matrix[0, 0], <size_t> matrix_size, <size_t> int(start_diagonal),
                                    <size_t> int(end_diagonal), &sum_table[0, 0])
    return sum_table

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_mean_table_2d_shuffled(
        numpy.ndarray[double, ndim=2, mode='c'] data_matrix not None,
        long start_diagonal,
        long end_diagonal):
    """
    """
    assert data_matrix.shape[0] == data_matrix.shape[1]
    matrix_size = data_matrix.shape[0]
    assert -matrix_size < start_diagonal < end_diagonal <= matrix_size

    cdef numpy.ndarray[double, ndim=2, mode='c'] mean_table = numpy.zeros(shape=(matrix_size, matrix_size), dtype=float,
                                                                          order='C')
    c_compute_mean_table_2d_shuffled(&data_matrix[0, 0], <size_t> matrix_size, <size_t> int(start_diagonal),
                                     <size_t> int(end_diagonal), &mean_table[0, 0])
    return mean_table

    
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_sum_table_1d_shuffled(
        numpy.ndarray[double, ndim=1, mode='c'] data_vector not None,
        long end_diagonal):
    """
    """
    vector_length = data_vector.shape[0]
    assert -vector_length < end_diagonal <= vector_length

    cdef numpy.ndarray[double, ndim=2, mode='c'] sum_table = numpy.zeros(shape=(vector_length, vector_length), dtype=float,
                                                                         order='C')
                                                                         
    c_compute_sum_table_1d_shuffled(&data_vector[0], <size_t> vector_length, 
                                    <size_t> int(end_diagonal), &sum_table[0, 0])
    return sum_table

    
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_mean_table_1d_shuffled(
        numpy.ndarray[double, ndim=1, mode='c'] data_vector not None,
        long end_diagonal):
    """
    """
    vector_length = data_vector.shape[0]
    assert -vector_length < end_diagonal <= vector_length

    cdef numpy.ndarray[double, ndim=2, mode='c'] mean_table = numpy.zeros(shape=(vector_length, vector_length), dtype=float,
                                                                          order='C')
    c_compute_mean_table_1d_shuffled(&data_vector[0], <size_t> vector_length, 
                                     <size_t> int(end_diagonal), &mean_table[0, 0])
    return mean_table
    
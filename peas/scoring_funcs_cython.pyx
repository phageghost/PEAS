# File: scoring_funcs_cython.pyx

import numpy
cimport numpy
import cython

cimport

numpy
import cython
import numpy

numpy.import_array()

cdef extern void c_scan_emissions(long*,
                                  double*,
                                  size_t,
                                  size_t,
                                  size_t,
                                  double*)

cdef extern void c_forward_backward_silent_native_sparse(size_t,
                                                         size_t*,
                                                         size_t*,
                                                         double*,
                                                         double*,
                                                         double*,
                                                         long*,
                                                         double*,
                                                         size_t,
                                                         size_t,
                                                         size_t)

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

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def scan_emissions(numpy.ndarray[long, ndim=1, mode='c'] observations not None,
                   numpy.ndarray[double, ndim=2, mode='c'] emissions not None):
    """
    Uses the emission matrix of an HMM to report likelihoods of the sequence starting at each location
    for an integer sequence of observations. Analogous to scanning a sequence using a PWM but not restricted
    to strings of nucleotides.
    """
    num_observations = observations.shape[0]
    num_states = emissions.shape[1]
    num_symbols = emissions.shape[0]

    cdef numpy.ndarray[double, ndim=1, mode='c'] scores = numpy.ones(shape=num_observations - num_states, dtype=float,
                                                                     order='C')

    c_scan_emissions(&observations[0],
                     &emissions[0, 0],
                     <size_t> num_states,
                     <size_t> num_symbols,
                     <size_t> num_observations,
                     &scores[0])

    return scores

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def forward_backward_silent_sparse(long num_submodels,
                                   numpy.ndarray[long, ndim=1, mode='c'] model_starts not None,
                                   numpy.ndarray[long, ndim=1, mode='c'] model_ends not None,
                                   numpy.ndarray[double, ndim=1, mode='c'] from_silent not None,
                                   numpy.ndarray[double, ndim=2, mode='c'] emissions not None,
                                   numpy.ndarray[double, ndim=1, mode='c'] forward_prior not None,
                                   numpy.ndarray[long, ndim=1, mode='c'] observations not None):
    """
    Cython shim for a C function that computes the smoothed estimate of state probability over a sequence
    of observations using the forward-backward algorithm.

    Parameters:
    :param:`num_submodels`
    :param:`model_starts`
    :param:`model_ends`
    :param:`from_silent`
    # :param:`to_silent`
    :param:`emissions`
    :param:`forward_prior`
    :param:`observations`

    Returns:
    A numpy array with |observations| rows and |states| columns giving the smoothed probability
    of each state at that point in the observation sequence.
    """

    num_observations = observations.shape[0]
    num_states = emissions.shape[1]
    num_symbols = emissions.shape[0]
    #print('{} observations, {} states, {} symbols'.format(num_observations, num_states, num_symbols))

    cdef numpy.ndarray[double, ndim=2, mode='c'] posterior_decoding = numpy.zeros(shape=(num_observations, num_states),
                                                                                  dtype=float, order='C')

    # c_forward_backward_silent_native_sparse(<size_t> num_submodels, <size_t*> &model_starts[0], <size_t*> &model_ends[0], &from_silent[0], &to_silent[0], &emissions[0,0], &forward_prior[0], &observations[0], &posterior_decoding[0,0], <size_t> num_states, <size_t> num_symbols, <size_t> num_observations)
    c_forward_backward_silent_native_sparse(<size_t> int(num_submodels), <size_t*> &model_starts[0],
                                            <size_t*> &model_ends[0], &from_silent[0], &emissions[0, 0],
                                            &forward_prior[0], &observations[0], &posterior_decoding[0, 0],
                                            <size_t> num_states, <size_t> num_symbols, <size_t> num_observations)

    # print('Back to Cython. Posterior decoding:')
    # print(posterior_decoding)

    return posterior_decoding

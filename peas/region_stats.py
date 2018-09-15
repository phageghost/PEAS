import datetime

import numpy

from peas.arrayfuncs import shuffle_matrix, my_diag_indices
from peas.fitapproxdistros.helper_funcs import fit_distros
from peas.utilities import log_print
from . import constants
from . import scoring
from . import scoring_funcs_cython


def generate_permuted_matrix_scores(matrix, num_shuffles, min_region_size=2, max_region_size=0, start_diagonal=1,
                                    matrix_score_func=scoring.compute_mean_table_2d,
                                    random_seed=None):
    """
    Given a matrix of values, returns a dictionary, keyed by region size, of
    scores (mean value) of regions of various size generated from shuffled
    copies of :param:`matrix`.
    """
    MIN_REPORTING_TIME = 5
    assert matrix.shape[0] == matrix.shape[1]
    log_print('setting random seed to {}'.format(random_seed), 3)
    numpy.random.seed(random_seed)
    n = matrix.shape[0]
    if max_region_size == 0:
        max_region_size = n

    sampled_scores = {region_size: numpy.empty((n - (region_size - 1)) * num_shuffles) for region_size in
                      range(min_region_size, max_region_size + 1)}

    last_time = datetime.datetime(1950, 1, 1)
    for shuffle_idx in range(num_shuffles):
        cur_time = datetime.datetime.now()
        elapsed_seconds = (cur_time - last_time).total_seconds()
        if elapsed_seconds > MIN_REPORTING_TIME:
            log_print('permutation {} of {}'.format(shuffle_idx + 1, num_shuffles), 3)
            last_time = cur_time

        if constants.USE_C:
            # ToDo: Clean up random seed stuff in C
            # ToDo: refactor to receive a scoring method name instead of a function pointer.
            # print('using C')
            scores = scoring_funcs_cython.compute_mean_table_2d_shuffled(data_matrix=matrix,
                                                                         start_diagonal=start_diagonal,
                                                                         end_diagonal=max_region_size)
                                                                         
        else:
            matrix = shuffle_matrix(matrix)
            scores = matrix_score_func(matrix, start_diagonal=start_diagonal, end_diagonal=max_region_size)

        for region_size in range(min_region_size, max_region_size + 1):
            diag_sample = numpy.diag(v=scores, k=region_size - 1)
            assert len(diag_sample) == n - (region_size - 1)

            sampled_scores[region_size][
            shuffle_idx * len(diag_sample):(shuffle_idx + 1) * len(diag_sample)] = diag_sample

    return sampled_scores


def fit_distributions(sampled_scores, support_ranges, matrix_size, start_diagonal,
                      distribution_class=constants.DEFAULT_DISTRO_CLASS,
                      parameter_smoothing_method=constants.DEFAULT_PARAMETER_SMOOTHING_METHOD,
                      parameter_smoothing_window_size=constants.SAVGOL_DEFAULT_WINDOW_SIZE):
    """
    Given a matrix of values, returns a dictionary, keyed by region size, of
    empirical distribution objects representing samples of scores of regions
    of that size taken from permuted versions of :param:`matrix`.
    """
    # ToDo: move and rename the parameter fitting and distribution generating functions.
    log_print('fitting distributions of class {}'.format(distribution_class), 2)
    # sizes = sorted(sampled_scores.keys())

    fit_params = fit_distros(sampled_scores, support_ranges=support_ranges, matrix_size=matrix_size,
                             start_diagonal=start_diagonal, distribution_class=distribution_class,
                             parameter_smoothing_window_size=parameter_smoothing_window_size)

    null_distributions = {}
    for region_size in fit_params:
        null_distributions[region_size] = distribution_class(*fit_params[region_size])

    return null_distributions


def compute_pscores(region_scores, null_distributions, tail):
    log_print('computing -log pvalues ...', 2)

    min_size = min(null_distributions.keys())
    max_size = max(null_distributions.keys())

    if len(region_scores.shape) == 1:  # ToDo: Do logsf with flexible tails to avoid this hackiness
        region_pvals = compute_pvalues_matrix(data_matrix=region_scores,
                                              distro_dict=null_distributions,
                                              diagonal_start=min_size - 1,
                                              diagonal_end=max_size,
                                              tail=tail)
        pval_scores = -numpy.log(numpy.maximum(region_pvals, constants.MIN_PVALUE))

    else:
        pval_scores = compute_pscores_matrix(data_matrix=region_scores,
                                             distro_dict=null_distributions,
                                             diagonal_start=min_size - 1,
                                             diagonal_end=max_size,
                                             tail=tail)

    return pval_scores


def compute_pvalues_tailed(frozen_distribution, x, tail='right'):
    if tail == 'right':
        return frozen_distribution.sf(x)
    elif tail == 'left':
        return frozen_distribution.cdf(x)
    elif tail == 'both':
        return 2 * numpy.minimum(frozen_distribution.cdf(x), frozen_distribution.sf(x))
    else:
        raise ValueError('Invalid value for parameter <tail>: {}'.format(tail))


def compute_log_pvals_tailed(frozen_distribution, x, tail='right'):
    if tail == 'right':
        return frozen_distribution.logsf(x)
    elif tail == 'left':
        return numpy.log(frozen_distribution.cdf(x))
    elif tail == 'both':
        return numpy.log(2 * numpy.minimum(frozen_distribution.cdf(x), frozen_distribution.sf(x)))
    else:
        raise ValueError('Invalid value for parameter <tail>: {}'.format(tail))


def compute_pvalues_matrix(data_matrix, distro_dict, tail='right', diagonal_start=1, diagonal_end=0, fill_value=1.0):
    """
    For every cell in :param:`data_matrix`, computes the p-value of the cell value using the corresponding distribution in :param:`distro_dict` where :math:`k` represents the distance of the cell from the matrix diagonal.

    """
    assert data_matrix.shape[0] == data_matrix.shape[1]
    n = data_matrix.shape[0]
    if diagonal_end == 0: diagonal_end = n

    p_vals = numpy.full((n, n), fill_value=fill_value, dtype=float)

    for diagonal in range(diagonal_start, diagonal_end):  # region size is diagonal + 1
        diag_indices = my_diag_indices(n, diagonal)

        p_vals[diag_indices] = compute_pvalues_tailed(frozen_distribution=distro_dict[diagonal + 1],
                                                      x=data_matrix[diag_indices], tail=tail)
    return p_vals


def compute_pscores_matrix(data_matrix, distro_dict, tail='right', diagonal_start=1, diagonal_end=0, fill_value=0.0):
    """
    For every cell in :param:`data_matrix`, computes the - log p-value of the cell value using the corresponding
     distribution instance  in :param:`distro_dict` where :math:`k` represents the distance of the cell from
    the matrix diagonal.

    """
    assert data_matrix.shape[0] == data_matrix.shape[1]
    n = data_matrix.shape[0]
    if diagonal_end == 0: diagonal_end = n

    p_scores = numpy.full((n, n), fill_value=fill_value, dtype=float)

    for diagonal in range(diagonal_start, diagonal_end):  # region size is diagonal + 1
        diag_indices = my_diag_indices(n, diagonal)

        p_scores[diag_indices] = -compute_log_pvals_tailed(frozen_distribution=distro_dict[diagonal + 1],
                                                           x=data_matrix[diag_indices], tail=tail)

    return p_scores


def compute_information_matrix(data_matrix, distro_dict, diagonal_start=1, diagonal_end=0, fill_value=0.0):
    """
    Computes the -log2 PDF for every cell in :param:`data_matrix`, using the
    distributions in :param:`distro_dict` corresponding to the region size.
    """
    assert data_matrix.shape[0] == data_matrix.shape[1]
    n = data_matrix.shape[0]
    if diagonal_end == 0: diagonal_end = n

    information_content = numpy.full((n, n), fill_value=fill_value, dtype=float)

    for diagonal in range(diagonal_start, diagonal_end):  # region size is diagonal + 1
        diag_indices = my_diag_indices(n, diagonal)
        pdf_results = distro_dict[diagonal + 1].pdf(data_matrix[diag_indices])
        information_content[diag_indices] = -numpy.log2(pdf_results)
    return information_content


def convert_pvals_to_pscores(pvals):
    return -numpy.log(pvals)


def convert_pscores_to_pvals(pscores):
    return numpy.exp(-pscores)

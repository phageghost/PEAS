import datetime

import numpy

import peas.fitapproxdistros.constants
from peas.arrayfuncs import my_diag_indices, truncate_array_tuple, shuffle_matrix
from peas.fitapproxdistros.helper_funcs import fit_distros
from peas.utilities import log_print

DEFAULT_DISTRO_CLASS = peas.fitapproxdistros.distributions.PiecewiseApproxLinear


def compute_sum_table_1d(vector, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the sum of the subarray
    of :param:`vector`centered bounded by the row and column indices

    Uses implicit recursion to do this efficiently.
    """
    n = len(vector)
    assert n > 0

    if end_diagonal == 0:
        end_diagonal = n

    sum_table = numpy.zeros((n, n))

    # Initialize: copy over the diagonal
    d0_idx = my_diag_indices(n, 0)
    sum_table[d0_idx] = vector

    if n > 1:
        # Second diagonal is left and beneath cells on 0th diagonal
        d1_idx = my_diag_indices(n, 1)
        #         print(d1_idx)
        sum_table[d1_idx] = sum_table[truncate_array_tuple(d0_idx, 1, 0)] + sum_table[
            truncate_array_tuple(d0_idx, 0, 1)]

        if n > 2:
            # 2rd to final diagonals are left and beneath cells on previous
            # diagonal, minus the diagonal left-down cell on the k-2 diagonal
            for k in range(2, end_diagonal):
                dk_idx = my_diag_indices(n, k)
                dk_prev = my_diag_indices(n, k - 1)
                dk_prevprev = my_diag_indices(n, k - 2)

                sum_table[dk_idx] = sum_table[truncate_array_tuple(dk_prev, 1, 0)] + sum_table[
                    truncate_array_tuple(dk_prev, 0, 1)] - sum_table[truncate_array_tuple(dk_prevprev, 1, 1)]

    return sum_table


def compute_denominator_1d(n):
    """
    Returns an (n X n) matrix containing the absolute
    value of the difference between the row index
    and the column index, plus one.

    Usefully,, this is the number of cells enclosed by
    the closed interval [row_index, column_index]
    """
    a = numpy.repeat(numpy.arange(0, n), n).reshape(n, n)
    b = a.T
    return numpy.abs(a - b) + 1


def compute_mean_table_1d(vector, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the mean of the subarray
    of :param:`data`centered bounded by the row and column indices

    Uses implicit recursion to do this efficiently.
    """
    n = len(vector)
    assert n > 0

    if end_diagonal == 0:
        end_diagonal = n

    mean_table = compute_sum_table_1d(vector=vector, end_diagonal=end_diagonal)

    ut_indices = numpy.triu_indices(n, 1)

    mean_table[ut_indices] /= compute_denominator_1d(n)[ut_indices]

    return mean_table


def compute_min_table_1d(vector, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the minimum of the subarray
    of :param:`vector`centered bounded by the row and column indices

    Uses implicit recursion to do this efficiently.
    """
    n = len(vector)
    assert n > 0

    if end_diagonal == 0:
        end_diagonal = n

    min_table = numpy.zeros((n, n))

    # Initialize: copy over the diagonal
    d0_idx = my_diag_indices(n, 0)
    min_table[d0_idx] = vector

    if n > 1:
        for k in range(1, end_diagonal):
            dk_idx = my_diag_indices(n, k)
            dk_prev = my_diag_indices(n, k - 1)
            min_table[dk_idx] = numpy.minimum(min_table[truncate_array_tuple(dk_prev, 1, 0)], min_table[
                truncate_array_tuple(dk_prev, 0, 1)])

    return min_table


def compute_max_table_1d(vector, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the maximum of the subarray
    of :param:`vector`centered bounded by the row and column indices

    Uses implicit recursion to do this efficiently.
    """
    n = len(vector)
    assert n > 0

    if end_diagonal == 0:
        end_diagonal = n

    max_table = numpy.zeros((n, n))

    # Initialize: copy over the diagonal
    d0_idx = my_diag_indices(n, 0)
    max_table[d0_idx] = vector

    if n > 1:
        for k in range(1, end_diagonal):
            dk_idx = my_diag_indices(n, k)
            dk_prev = my_diag_indices(n, k - 1)
            max_table[dk_idx] = numpy.maximum(max_table[truncate_array_tuple(dk_prev, 1, 0)], max_table[
                truncate_array_tuple(dk_prev, 0, 1)])

    return max_table


def compute_sum_table_2d(data, start_diagonal=0, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the sum of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently.
    """
    assert data.shape[0] == data.shape[1]
    n = data.shape[0]
    assert n > 0
    assert n >= end_diagonal

    if end_diagonal == 0:
        end_diagonal = n

    assert end_diagonal - start_diagonal > 0

    sum_table = numpy.zeros((n, n))

    # Initialize: copy over the 1st diagonal
    d1_idx = my_diag_indices(n, start_diagonal)
    sum_table[d1_idx] = data[d1_idx]

    # Second diagonal is contents of second diagonal plus left and beneath cells on 1st diagonal
    if end_diagonal - start_diagonal >= 2:
        d2_idx = my_diag_indices(n, start_diagonal + 1)
        sum_table[d2_idx] = data[d2_idx] + sum_table[truncate_array_tuple(d1_idx, 1, 0)] + sum_table[
            truncate_array_tuple(d1_idx, 0, 1)]

        if end_diagonal - start_diagonal >= 3:
            # 3rd to final diagonals are contents of that diagonal plus left and beneath cells on previous
            # diagonal, minus the diagonal left-down cell on the k-2 diagonal
            for diagonal_idx in range(start_diagonal + 2, end_diagonal):
                dk_idx = my_diag_indices(n, diagonal_idx)
                dk_prev = my_diag_indices(n, diagonal_idx - 1)
                dk_prevprev = my_diag_indices(n, diagonal_idx - 2)

                sum_table[dk_idx] = data[dk_idx] + sum_table[truncate_array_tuple(dk_prev, 1, 0)] + sum_table[
                    truncate_array_tuple(dk_prev, 0, 1)] - sum_table[truncate_array_tuple(dk_prevprev, 1, 1)]

    return sum_table


def compute_denominator_2d(n, start_diagonal=1):
    """
    Returns an upper-triangular matrix containing the number of cells
    in a triangular region around the diagonal with a corner at that cell, excluding
    cells on diagonals smaller than :param start_diagonal:.

    Used for converting tables of region sums to means.
    """
    assert n >= 1
    assert start_diagonal >= 0

    denom_table = numpy.zeros((n, n))
    if start_diagonal == 0:
        dk_idx = my_diag_indices(n, 0)
        denom_table[dk_idx] = 1

    for k in range(max(start_diagonal, 1), n):
        dk_idx = my_diag_indices(n, k)
        dk_prev_idx = truncate_array_tuple(my_diag_indices(n, k - 1), 0, 1)
        denom_table[dk_idx] = denom_table[dk_prev_idx] + numpy.full(n - k, fill_value=k - start_diagonal + 1)

    return denom_table


def compute_mean_table_2d(data, start_diagonal=0, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the mean of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently.
    """
    assert data.shape[0] == data.shape[1]
    n = data.shape[0]
    assert n > 0
    assert n >= end_diagonal

    if end_diagonal == 0:
        end_diagonal = n

    assert end_diagonal - start_diagonal > 0

    mean_table = compute_sum_table_2d(data, start_diagonal=start_diagonal, end_diagonal=end_diagonal)
    ut_indices = numpy.triu_indices(n, start_diagonal)

    mean_table[ut_indices] /= compute_denominator_2d(n, start_diagonal=start_diagonal)[ut_indices]

    return mean_table


def compute_min_table_2d(data, start_diagonal=0, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the sum of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently.
    """
    assert data.shape[0] == data.shape[1]
    n = data.shape[0]
    assert n > 0
    assert n >= end_diagonal

    if end_diagonal == 0:
        end_diagonal = n

    assert end_diagonal > start_diagonal

    min_table = numpy.zeros((n, n))

    # Initialize: copy over the 1st diagonal
    d1_idx = my_diag_indices(n, start_diagonal)
    min_table[d1_idx] = data[d1_idx]
    if end_diagonal - start_diagonal >= 2:
        for diagonal_idx in range(start_diagonal + 1, end_diagonal):
            dk_idx = my_diag_indices(n, diagonal_idx)
            dk_prev = my_diag_indices(n, diagonal_idx - 1)

            min_table[dk_idx] = numpy.minimum(data[dk_idx],
                                              numpy.minimum(min_table[truncate_array_tuple(dk_prev, 1, 0)],
                                                            min_table[truncate_array_tuple(dk_prev, 0, 1)]))
    return min_table


def compute_max_table_2d(data, start_diagonal=0, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the sum of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently.
    """
    assert data.shape[0] == data.shape[1]
    n = data.shape[0]
    assert n > 0
    assert n >= end_diagonal

    if end_diagonal == 0:
        end_diagonal = n

    assert end_diagonal > start_diagonal

    max_table = numpy.zeros((n, n))

    # Initialize: copy over the 1st diagonal
    d1_idx = my_diag_indices(n, start_diagonal)
    max_table[d1_idx] = data[d1_idx]
    if end_diagonal - start_diagonal >= 2:
        for diagonal_idx in range(start_diagonal + 1, end_diagonal):
            dk_idx = my_diag_indices(n, diagonal_idx)
            dk_prev = my_diag_indices(n, diagonal_idx - 1)

            max_table[dk_idx] = numpy.maximum(data[dk_idx],
                                              numpy.maximum(max_table[truncate_array_tuple(dk_prev, 1, 0)],
                                                            max_table[truncate_array_tuple(dk_prev, 0, 1)]))
    return max_table


def generate_permuted_matrix_scores(matrix, num_shuffles, min_region_size=2, max_region_size=0, start_diagonal=1,
                                    matrix_score_func=compute_mean_table_2d,
                                    random_seed=None):
    """
    Given a matrix of values, returns a dictionary, keyed by region size, of
    scores (mean value) of regions of various size generated from shuffled
    copies of :param:`matrix`.
    """
    MIN_REPORTING_TIME = 5
    assert matrix.shape[0] == matrix.shape[1]
    log_print('Setting random seed to {}'.format(random_seed), 3)
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
            log_print('permutation {} of {}'.format(shuffle_idx + 1, num_shuffles), 4)
            last_time = cur_time
        matrix = shuffle_matrix(matrix)
        scores = matrix_score_func(matrix, start_diagonal=start_diagonal)
        for region_size in range(min_region_size, max_region_size + 1):
            diag_sample = numpy.diag(v=scores, k=region_size - 1)
            assert len(diag_sample) == n - (region_size - 1)

            sampled_scores[region_size][
            shuffle_idx * len(diag_sample):(shuffle_idx + 1) * len(diag_sample)] = diag_sample

    return sampled_scores


# ToDo: split function to only perform fitting.
def fit_distributions(sampled_scores, distribution_class=peas.fitapproxdistros.constants.DEFAULT_DISTRO_CLASS,
                      parameter_smoothing_method=peas.fitapproxdistros.constants.DEFAULT_PARAMETER_SMOOTHING_METHOD,
                      parameter_smoothing_window_size=peas.fitapproxdistros.constants.SAVGOL_DEFAULT_WINDOW_SIZE):
    """
    Given a matrix of values, returns a dictionary, keyed by region size, of
    empirical distribution objects representing samples of scores of regions
    of that size taken from permuted versions of :param:`matrix`.
    """
    log_print('Fitting distributions of class {}'.format(distribution_class), 2)
    sizes = sorted(sampled_scores.keys())

    fit_params = fit_distros(sampled_scores, distribution_class=distribution_class,
                             parameter_smoothing_window_size=parameter_smoothing_window_size)

    empirical_distros = {}
    for region_size in sizes:
        empirical_distros[region_size] = distribution_class(*fit_params[region_size])

    return empirical_distros


def compute_pscores(region_scores, empirical_distros, tail):
    log_print('computing -log pvalues ...', 2)

    if len(region_scores.shape) == 1:  # ToDo: Do logsf with flexible tails to avoid this hackiness
        region_pvals = simstraps.compute_pvalues_matrix(data_matrix=region_scores,
                                                        distro_dict=empirical_distros,
                                                        diagonal_start=min_size - 1,
                                                        diagonal_end=max(empirical_distros.keys()),
                                                        tail=tail)
        pval_scores = -numpy.log(numpy.maximum(region_pvals, MIN_PVALUE))

    else:
        pval_scores = simstraps.compute_pscores_matrix(data_matrix=region_scores,
                                                       distro_dict=empirical_distros,
                                                       diagonal_start=min_size - 1,
                                                       diagonal_end=max_distro_size,
                                                       tail=tail)

    return pval_scores


def compute_pvals(frozen_distribution, x, tail='right'):
    if tail == 'right':
        return frozen_distribution.sf(x)
    elif tail == 'left':
        return frozen_distribution.cdf(x)
    elif tail == 'both':
        return 2 * numpy.minimum(frozen_distribution.cdf(x), frozen_distribution.sf(x))
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
        frozen_distribution = distro_dict[diagonal + 1]

        p_vals[diag_indices] = compute_pvals(frozen_distribution=distro_dict[diagonal + 1],
                                             x=data_matrix[diag_indices], tail=tail)
    return p_vals


def compute_pscores_matrix(data_matrix, distro_dict, tail='right', diagonal_start=1, diagonal_end=0, fill_value=0.0):
    """
    For every cell in :param:`data_matrix`, computes the p-value of the cell value using the corresponding distribution in :param:`distro_dict` where :math:`k` represents the distance of the cell from the matrix diagonal.

    """
    assert data_matrix.shape[0] == data_matrix.shape[1]
    n = data_matrix.shape[0]
    if diagonal_end == 0: diagonal_end = n

    p_scores = numpy.full((n, n), fill_value=fill_value, dtype=float)

    for diagonal in range(diagonal_start, diagonal_end):  # region size is diagonal + 1
        diag_indices = my_diag_indices(n, diagonal)
        # normal approximation:
        frozen_distribution = distro_dict[diagonal + 1]
        # m = frozen_distribution.data.mean()
        # s = frozen_distribution.data.std()
        # p_scores[diag_indices] = -scipy.stats.norm(m,s).logsf(data_matrix[diag_indices])
        p_scores[diag_indices] = -frozen_distribution.logsf(data_matrix[diag_indices])

        # p_vals[diag_indices] = compute_pvals(frozen_distribution=distro_dict[diagonal + 1],
        # x=data_matrix[diag_indices], tail=tail)
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

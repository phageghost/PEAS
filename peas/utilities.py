import datetime

import numpy
import scipy.stats


def pretty_now():
    """
    Returns the current date/time in a nicely formatted string (without so many decimal places)
    """
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%b-%d %H:%M:%S')


def log_print(message, tabs=1):
    print('{}{}{}'.format(pretty_now(), '\t' * tabs, message))


def validate_param(param_name, value_received, allowable_values):
    assert value_received in allowable_values, 'Received invalid value \'{}\' for parameter {}. Allowable values: {}'.format(
        value_received, param_name, ', '.join(allowable_values))


def my_diag_indices(n, k=0):
    """
    Return the indices corresponding to the kth diagonal of an n X n array
    in the form of a tuple of (x coords, y coords).

    Created since numpy does not provide this function.
    """
    if k <= 0:
        x_coords = numpy.arange(-k, n)
        y_coords = numpy.arange(0, n + k)
    else:
        x_coords = numpy.arange(0, n - k)
        y_coords = numpy.arange(k, n)

    return (x_coords, y_coords)


def truncate_array_tuple(array_tuple, prefix_trim, suffix_trim):
    """
    Given a tuple of arrays, trim :param:`prefix_trim` elements from
    the beginning and :param:`suffix_trim` elements from the end of
    each array and return the result.
    """
    if prefix_trim > 0 and suffix_trim > 0:
        return tuple([arr[prefix_trim:-suffix_trim] for arr in array_tuple])
    if prefix_trim > 0:
        return tuple([arr[prefix_trim:] for arr in array_tuple])
    if suffix_trim > 0:
        return tuple([arr[:-suffix_trim] for arr in array_tuple])
    return array_tuple


def resample_array(arr, new_size, support):
    """
    Return an interpolated copy of size :param new_size: of :param arr: with size given
    as the span of :param support:.
    :param arr:
    :param new_size:
    :param support:
    :return:
    """
    resampled = numpy.interp(x=numpy.linspace(*support, new_size), xp=numpy.linspace(*support, len(arr)), fp=arr)
    resampled /= resampled.sum()
    return resampled


def clean_array(arr):
    """
    Returns a copy of :param:`arr` with all inf, neginf and NaN values removed
    """
    return arr[numpy.nonzero(~(numpy.isnan(arr) | numpy.isinf(arr) | numpy.isneginf(arr)))[0]]


def replace_nans_diagonal_means(matrix, start_diagonal=0, end_diagonal=0):
    """
    Returns a copy of :param:`matrix` where all NaN values are replaced
    by the mean of that cell's diagonal vector (computed without NaNs).

    Requires that no diagonals consist only of NaNs (run trim_matrix_edges first)
    """
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]
    if end_diagonal == 0:
        end_diagonal = n - 1
        start_diagonal = -end_diagonal

    filled_matrix = matrix.copy()
    for diag_idx in range(start_diagonal, end_diagonal):
        diag_indices = my_diag_indices(n, diag_idx)
        diag_vector = matrix[diag_indices]
        bad_locs = numpy.isnan(diag_vector)
        good_locs = numpy.logical_not(bad_locs)
        diag_mean = diag_vector[good_locs].mean()
        diag_vector[bad_locs] = diag_mean
        filled_matrix[diag_indices] = diag_vector
    return filled_matrix


def compute_matrix_trim_points(x):
    """
    Returns a 4-tuple for the following coordinates needed to trim :param:`x`
    so that all edge rows and columns that contain no valid entries are removed.
    """
    # rows
    nan_rows = (numpy.isnan(x).sum(axis=1) == x.shape[0]).astype(int)
    row_transitions = numpy.diff(nan_rows)

    row_candidate_start_trim_points = numpy.nonzero(row_transitions < 0)[0]
    if nan_rows[0] == 1 and len(row_candidate_start_trim_points) > 0:
        row_start_trim_point = row_candidate_start_trim_points[0] + 1
    else:
        row_start_trim_point = 0

    row_candidate_end_trim_points = numpy.nonzero(row_transitions > 0)[0]
    if nan_rows[-1] == 1 and len(row_candidate_end_trim_points) > 0:
        row_end_trim_point = row_candidate_end_trim_points[-1]
    else:
        row_end_trim_point = x.shape[0]

    # cols
    nan_cols = (numpy.isnan(x).sum(axis=0) == x.shape[1]).astype(int)
    col_transitions = numpy.diff(nan_cols)

    col_candidate_start_trim_points = numpy.nonzero(col_transitions < 0)[0]
    if nan_cols[0] == 1 and len(col_candidate_start_trim_points) > 0:
        col_start_trim_point = col_candidate_start_trim_points[0] + 1
    else:
        col_start_trim_point = 0

    col_candidate_end_trim_points = numpy.nonzero(col_transitions > 0)[0]
    if nan_cols[-1] == 1 and len(col_candidate_end_trim_points) > 0:
        col_end_trim_point = col_candidate_end_trim_points[-1] + 1
    else:
        col_end_trim_point = x.shape[1]

    return row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point


def gaussian_kernel(sd, sd_cutoff=3, normalize=False):
    """
    Return an array containing discrete samples from a continuous Gaussian curve
    at integer points in the interval (-:param sd: * sd_cutoff, +:param sd: * sd_cutoff).

    If :param normalize: is True, the peak of the curve will be 1.0, otherwise it the
    values will be that of a Normal PDF having mean 0 and standard deviation :param sd:.

    :param sd:
    :param sd_cutoff:
    :param normalize:
    :return:
    """
    bw = int(sd_cutoff * sd * 2 + 1)
    midpoint = sd_cutoff * sd
    kern = numpy.zeros(bw)
    frozen_rv = scipy.stats.norm(scale=sd)
    for i in range(bw):
        kern[i] = frozen_rv.pdf(i - midpoint)
    if normalize:
        kern = kern / kern.max()
    return kern


def quantiles(data):
    """
    Returns a pandas Series of the quantiles of data in <data>. Quantiles start at 1 / (len(data) + 1) and
    end at len(data) / (len(data) + 1) to avoid singularities at the 0 and 1 quantiles.
    to prevent
    :param data:
    :return:
    """
    sort_indices = numpy.argsort(data)
    quants = pandas.Series(numpy.zeros(len(data)))
    try:
        quants.index = data.index
    except AttributeError:
        pass
    quants[sort_indices] = (numpy.arange(len(data)) + 1) / float(len(data) + 1)
    return quants


def gaussian_norm(arr):
    """
    Quantile normalizes the given array to a standard Gaussian distribution
    :param data:
    :return:
    """
    quants = numpy.array(quantiles(arr))
    std_normal = scipy.stats.norm(loc=0, scale=1)
    normed = std_normal.ppf(quants)

    return normed

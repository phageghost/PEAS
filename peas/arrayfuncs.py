import numpy


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

    return x_coords, y_coords


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


def shuffle_matrix(matrix):
    """
    Returns a randomly huffled copy of 2D array :param matrix: that preserves row and column affiliations

    :param matrix:
    :return:
    """
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]

    shuff_indices = numpy.arange(n)
    numpy.random.shuffle(shuff_indices)

    return matrix[shuff_indices][:, shuff_indices]


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


def compute_vector_trim_points(arr):
    """
    Returns the start and end coordinates of a maximum-size central non-NaN subarray of :param arr:

    :param arr:
    :return:
    """
    # ToDo Extend to inf and neginf
    nan_elements = numpy.isnan(arr).astype(int)
    transitions = numpy.diff(nan_elements)

    candidate_start_trim_points = numpy.nonzero(transitions < 0)[0]
    if nan_elements[0] == 1 and len(candidate_start_trim_points) > 0:
        trim_start = candidate_start_trim_points[0] + 1
    else:
        trim_start = 0

    candidate_end_trim_points = numpy.nonzero(transitions > 0)[0]
    if nan_elements[-1] == 1 and len(candidate_end_trim_points) > 0:
        trim_end = candidate_end_trim_points[-1] + 1
    else:
        trim_end = len(arr)

    return trim_start, trim_end


def compute_matrix_trim_points(arr):
    """
    Returns a 4-tuple for the following coordinates needed to trim :param:`x`
    so that all edge rows and columns that contain no valid entries are removed.
    """
    # ToDo Extend to inf and neginf
    # rows
    nan_rows = (numpy.isnan(arr).sum(axis=1) == arr.shape[0]).astype(int)
    row_transitions = numpy.diff(nan_rows)

    row_candidate_start_trim_points = numpy.nonzero(row_transitions < 0)[0]
    if nan_rows[0] == 1 and len(row_candidate_start_trim_points) > 0:
        row_start_trim_point = row_candidate_start_trim_points[0] + 1
    else:
        row_start_trim_point = 0

    row_candidate_end_trim_points = numpy.nonzero(row_transitions > 0)[0]
    if nan_rows[-1] == 1 and len(row_candidate_end_trim_points) > 0:
        row_end_trim_point = row_candidate_end_trim_points[-1] + 1
    else:
        row_end_trim_point = arr.shape[0]

    # cols
    nan_cols = (numpy.isnan(arr).sum(axis=0) == arr.shape[1]).astype(int)
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
        col_end_trim_point = arr.shape[1]

    return row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point


def create_diagonal_distance_matrix(n):
    """
    Returns an :param:`n` X :param:`n` matrix containing the distance from the diagonal in each cell.
    """
    return numpy.repeat(numpy.arange(n).reshape(n, 1), n, axis=1)[::-1] + numpy.repeat(numpy.arange(n).reshape(1, n), n,
                                                                                       axis=0) - numpy.full((n, n),
                                                                                                            n - 1)


def create_data_masks(mask_2d):
    """
    Given a 2-dimensional boolean array indicating which cells have passed,
    returns two lists: a list of numpy arrays giving the x-coordinates of
    passing cells for the ith row and a list of numpy arrays giving the
    y-coordinates of passing cells for the ith column.
    """
    assert mask_2d.shape[0] == mask_2d.shape[1]
    n = mask_2d.shape[0]

    row_masks = []
    col_masks = []

    for i in range(n):
        row_masks.append(numpy.nonzero(mask_2d[i, i:])[0] + i)
        col_masks.append(numpy.nonzero(mask_2d[:i, i])[0])

    return row_masks, col_masks

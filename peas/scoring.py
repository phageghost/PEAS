import numpy

from peas.arrayfuncs import my_diag_indices, truncate_array_tuple


def compute_sum_table_2d(data, start_diagonal=1, end_diagonal=0):
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


def compute_mean_table_2d(data, start_diagonal=1, end_diagonal=0):
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

    mean_table[ut_indices] /= compute_dp_denominator(n, start_diagonal=start_diagonal)[ut_indices]

    return mean_table


def compute_dp_denominator(n, start_diagonal=1):
    """
    Returns an upper-triangular matrix containing the number of cells
    in a square with a corner at that cell, centered on the diagonal.
    """
    assert n > 0

    denom_table = numpy.zeros((n, n))
    for k in range(start_diagonal, n):
        dk_idx = my_diag_indices(n, k)
        dk_prev_idx = truncate_array_tuple(my_diag_indices(n, k - 1), 0, 1)
        # print(k, dk_idx, dk_prev_idx)
        denom_table[dk_idx] = denom_table[dk_prev_idx] + numpy.full(n - k, fill_value=k - start_diagonal + 1)
    return denom_table


def compute_1d_denominator(n):
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


def compute_sum_table_1d(data_vec, start_diagonal=0, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the mean of the subarray
    of :param:`data`centered bounded by the row and column indices

    Uses implicit recursion to do this efficiently.
    """
    n = len(data_vec)
    assert n > 0

    if end_diagonal == 0:
        end_diagonal = n

    sum_table = numpy.zeros((n, n))

    # Initialize: copy over the diagonal
    d0_idx = my_diag_indices(n, 0)
    sum_table[d0_idx] = data_vec

    if n > 1:
        # Second diagonal is left and beneath cells on 0th diagonal
        d1_idx = my_diag_indices(n, 1)
        #         print(d1_idx)
        sum_table[d1_idx] = sum_table[truncate_array_tuple(d0_idx, 1, 0)] + sum_table[
            truncate_array_tuple(d0_idx, 0, 1)]

        if n > 2:
            # 2rd to final diagonals are left and beneath cells on previous
            # diagonal, minus the diagonal left-down cell on the k-2 diagonal
            for k in range(2, n):
                dk_idx = my_diag_indices(n, k)
                dk_prev = my_diag_indices(n, k - 1)
                dk_prevprev = my_diag_indices(n, k - 2)

                sum_table[dk_idx] = sum_table[truncate_array_tuple(dk_prev, 1, 0)] + sum_table[
                    truncate_array_tuple(dk_prev, 0, 1)] - sum_table[truncate_array_tuple(dk_prevprev, 1, 1)]

    return sum_table


def compute_mean_table_1d(data_vec, start_diagonal=0, end_diagonal=0):
    """
    Returns an upper-triangular matrix where each cell contains the mean of the subarray
    of :param:`data`centered bounded by the row and column indices

    Uses implicit recursion to do this efficiently.
    """
    n = len(data_vec)
    assert n > 0

    if end_diagonal == 0:
        end_diagonal = n

    mean_table = compute_sum_table_1d(data_vec=data_vec, start_diagonal=start_diagonal, end_diagonal=end_diagonal)

    ut_indices = numpy.triu_indices(n, 1)

    mean_table[ut_indices] /= compute_1d_denominator(n)[ut_indices]

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

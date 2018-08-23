import numpy
import scipy.stats


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


def clean_array(arr):
    """
    Returns a copy of :param:`arr` with all inf, neginf and NaN values removed
    """
    return arr[numpy.nonzero(~(numpy.isnan(arr) | numpy.isinf(arr) | numpy.isneginf(arr)))[0]]


def resample_array(arr, new_size, support):
    """
    """
    resampled = numpy.interp(x=numpy.linspace(*support, new_size), xp=numpy.linspace(*support, len(arr)), fp=arr)
    resampled /= resampled.sum()
    return resampled


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

import datetime
import sys

import numpy
import pandas
import scipy.stats


def pretty_now():
    """
    Returns the current date/time in a nicely formatted string (without so many decimal places)
    """
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%b-%d %H:%M:%S')


def eprint(*args, **kwargs):
    """
    Print to stderr.
    """
    print(*args, file=sys.stderr, **kwargs)


def log_print(message, tabs=1):
    eprint('{}{}{}'.format(pretty_now(), '\t' * tabs, message))


def validate_param(param_name, value_received, allowable_values):
    result = value_received in allowable_values
    assert result, 'Received invalid value \'{}\' for parameter {}. Allowable values: {}'.format(
        value_received, param_name, ', '.join(allowable_values))
    return result


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


def force_odd(num):
    if num % 2 == 0:
        num += 1
    return num


def force_even(num):
    if num % 2 == 1:
        num += 1
    return num

    
def pseudolinearize(arr, pseudocount=1):
    return (numpy.log2(arr + pseudocount) - numpy.log2(pseudocount))    
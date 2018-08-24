import datetime

import numpy
import scipy.stats
from empdist import EmpiricalDistribution
from scipy.signal import convolve

from peas.arrayfuncs import resample_array, my_diag_indices
from peas.utilities import log_print

DEFAULT_MAX_EMPIRICAL_SIZE = 50
SAVGOL_DEFAULT_WINDOW_SIZE = 5


def predict_distributions_independent_sums(input_empirical_distribution,
                                           max_sample_size):
    """
    Given a population sample, return a list of frozen empirical distribution
    objects representing the inferred distributions of the sum of samples taken from
    that population of sizes [1, :param:`max_sample_size`].

    Returns a dictionary, keyed by region size, of EmpiricalDistribution objects.
    """
    assert max_sample_size >= 1

    empirical_distros_by_region_size = {1: input_empirical_distribution}

    for sample_size in range(2, max_sample_size + 1):
        empirical_distros_by_region_size[sample_size] = empirical_distros_by_region_size[
                                                            sample_size - 1] + input_empirical_distribution

    return empirical_distros_by_region_size






























def generate_empirical_distributions_region_means(data,
                                                  max_region_size,
                                                  bins='auto',
                                                  pseudocount=0):
    """
    Given a population sample, return a list of frozen empirical distribution
    objects representing the inferred distributions of the mean value of samples taken from
    that population of sizes [1, :param:`max_region_size`].

    Distributions of sample sizes below :param:`max_empirical_size` will be empirical piecewise
    histograms of size :param:`num_bins` * sample_size.

    Distributions of sample sizes above :param:`max_empirical_size` will be normal approximations.

    Returns a dictionary, keyed by region size, of MyEmpiricalDistribution objects.
    """
    # ToDo: Clean up flow control.
    assert max_region_size >= 1
    empirical_distros_by_region_size = {}

    if len(data.shape) > 1:  # force data to be 1-D
        data = data.flatten()

    data_mean, data_var = data.mean(), data.var()
    if support is None:
        support = (data.min(), data.max())

    if max_empirical_size < 1:
        pdfs = [scipy.stats.norm(loc=data_mean, scale=numpy.sqrt(data_var))]
    else:
        singleton_pdf = EmpiricalDistribution.from_data(data, bins=bins, pseudocount=pseudocount,
                                                        support=support)
        new_frequencies = singleton_pdf.frequencies
        pdfs = [singleton_pdf]

    for region_size in range(2, min(max_region_size, max_empirical_size) + 1):
        new_frequencies = scipy.signal.convolve(new_frequencies, singleton_pdf.frequencies, mode='full', method='auto')

        if resample:
            empirical_distros_by_region_size[region_size] = EmpiricalDistribution \
                (resample_array(new_frequencies, new_size=num_bins, support=support), support=support,
                 pseudocount=pseudocount)
        else:
            empirical_distros_by_region_size[region_size] = EmpiricalDistribution(new_frequencies, support=support,
                                                                                  pseudocount=pseudocount)

    if max_region_size > max_empirical_size:
        for region_size in range(max(2, max_empirical_size), max_region_size + 1):
            new_pdf = scipy.stats.norm(loc=data_mean, scale=numpy.sqrt(data_var * region_size) / region_size)
            empirical_distros_by_region_size[region_size] = new_pdf

    return empirical_distros_by_region_size


def resample_region_means(matrix, num_shuffles, min_region_size=2, max_region_size=0, start_diagonal=1,
                          random_seed=None):
    """
    Given a matrix of values, returns a dictionary, keyed by region size, of
    scores (mean value) of regions of various size generated from shuffled
    copies of :param:`matrix`.
    """
    MIN_REPORTING_TIME = 1
    assert matrix.shape[0] == matrix.shape[1]
    log_print('Setting random seed to {}'.format(random_seed), 3)
    numpy.random.seed(random_seed)
    n = matrix.shape[0]
    if max_region_size == 0:
        max_region_size = n

    diag_indices = {region_size: my_diag_indices(n, k=region_size - 1) for region_size in
                    range(min_region_size, max_region_size + 1)}
    sampled_scores = {region_size: numpy.empty((n - (region_size - 1)) * num_shuffles) for region_size in
                      range(min_region_size, max_region_size + 1)}
    sample_indices = {region_size: 0 for region_size in range(min_region_size, max_region_size + 1)}

    # print('n {} start diagonal {}'.format(n, start_diagonal))
    upper_tri_indices = numpy.triu_indices(n, start_diagonal)

    # log_print('min: {}, mean: {}, max: {}'.format(matrix[upper_tri_indices].min(), matrix[upper_tri_indices].mean(), matrix[upper_tri_indices].max()), 4)

    last_time = datetime.datetime(1950, 1, 1)
    for shuffle_idx in range(num_shuffles):
        cur_time = datetime.datetime.now()
        elapsed_seconds = (cur_time - last_time).total_seconds()
        if elapsed_seconds > MIN_REPORTING_TIME:
            log_print('permutation {} of {}'.format(shuffle_idx + 1, num_shuffles), 4)
            last_time = cur_time
        matrix = shuffle_matrix(matrix)
        # log_print('min: {}, mean: {}, max: {}'.format(matrix[upper_tri_indices].min(), matrix[upper_tri_indices].mean(), matrix[upper_tri_indices].max()), 4)
        mean_matrix = compute_mean_table_2d(matrix, start_diagonal=start_diagonal)
        for region_size in range(min_region_size, max_region_size + 1):
            diag_sample = mean_matrix[diag_indices[region_size]]
            sampled_scores[region_size][sample_indices[region_size]:sample_indices[region_size] + len(diag_sample)] = \
                mean_matrix[diag_indices[region_size]]
            sample_indices[region_size] += len(diag_sample)
    return sampled_scores


def smooth_parameters(param_dict, filter_window_size=SAVGOL_DEFAULT_WINDOW_SIZE):
    # ToDo: Refactor for elegance
    if len(param_dict) >= 3:
        if filter_window_size:
            filter_window_size = max(force_odd(int(filter_window_size)), 3)
            log_print('Smoothing parameters with Savitsky-Golay filter of size {}'.format(filter_window_size), 3)
            param_df = pandas.DataFrame(param_dict).T
            param_array = scipy.signal.savgol_filter(param_df, filter_window_size, 1, axis=0)
            param_dict = {region_size: params for region_size, params in
                          zip(sorted(param_dict.keys()), param_array.tolist())}
    else:
        log_print('Too few region sizes to perform parameter smoothing (need at least 3)', 3)

    return param_dict


def fit_distros(shuffled_samples, distro_class, filter_window_size=SAVGOL_DEFAULT_WINDOW_SIZE):
    """
    Given a dictionary of permuted data vectors, return a dictionary of optimal parameters
    (as tuples) for distributions of class :param:`distro_class`.
    """
    sizes = sorted(shuffled_samples.keys())

    fit_params = {}
    for region_size in sizes:
        # log_print('size {}, min score: {}, mean score: {}, max score: {}'.format(region_size, sampled_scores[region_size].min(), sampled_scores[region_size].mean(), sampled_scores[region_size].max()),3)
        this_fit_params = distro_class.fit(shuffled_samples[region_size])
        fit_params[region_size] = this_fit_params
        log_print('size: {} fit parameters: {}'.format(region_size, this_fit_params), 3)

    return smooth_parameters(fit_params, filter_window_size=filter_window_size)


def generate_empirical_distributions_dependent_region_means(matrix, num_shuffles, min_region_size=2, max_region_size=0,
                                                            start_diagonal=1, random_seed=None,
                                                            distro_class=DEFAULT_DISTRO_CLASS,
                                                            filter_window_size=SAVGOL_DEFAULT_WINDOW_SIZE):
    """
    Given a matrix of values, returns a dictionary, keyed by region size, of
    empirical distribution objects representing samples of scores of regions
    of that size taken from permuted versions of :param:`matrix`.
    """
    sampled_scores = resample_region_means(matrix, num_shuffles, min_region_size, max_region_size, start_diagonal,
                                           random_seed=random_seed)

    log_print('Fitting distributions of class {}'.format(distro_class), 2)
    sizes = sorted(sampled_scores.keys())

    fit_params = fit_distros(sampled_scores, distro_class, filter_window_size=filter_window_size)

    empirical_distros = {}
    for region_size in sizes:
        empirical_distros[region_size] = distro_class(*fit_params[region_size])

    return empirical_distros

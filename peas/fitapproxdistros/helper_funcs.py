import pandas
import scipy.stats
from scipy.signal import savgol_filter

from peas.utilities import log_print, force_odd
from . import constants


def smooth_parameters(param_dict, parameter_smoothing_window_size=constants.SAVGOL_DEFAULT_WINDOW_SIZE):
    # ToDo: Refactor for elegance
    if len(param_dict) >= 3:
        if parameter_smoothing_window_size:
            parameter_smoothing_window_size = max(force_odd(int(parameter_smoothing_window_size)), 3)
            log_print(
                'smoothing parameters with Savitsky-Golay filter of size {}'.format(parameter_smoothing_window_size), 3)
            param_df = pandas.DataFrame(param_dict).T  # ToDo: refactor to remove pandas dependency here.
            param_array = savgol_filter(param_df, parameter_smoothing_window_size, 1, axis=0)
            param_dict = {region_size: params for region_size, params in
                          zip(sorted(param_dict.keys()), param_array.tolist())}
    else:
        log_print('Too few region sizes to perform parameter smoothing (need at least 3)', 3)

    return param_dict


def fit_distros(shuffled_samples, distribution_class,
                support_ranges,
                matrix_size,
                start_diagonal=1,
                max_pvalue_std_error=constants.DEFAULT_MAX_PVALUE_SE,
                parameter_smoothing_method=None,
                parameter_smoothing_window_size=constants.SAVGOL_DEFAULT_WINDOW_SIZE, fit_kwargs={}):
    """
    Given a dictionary of permuted data vectors, return a dictionary of optimal parameters
    (as tuples) for distributions of class :param:`distro_class`.
    """
    region_sizes = sorted(shuffled_samples.keys())

    fit_params = {}
    for region_size in region_sizes:
        # log_print('size {}, min score: {}, mean score: {}, max score: {}'.format(region_size, sampled_scores[region_size].min(), sampled_scores[region_size].mean(), sampled_scores[region_size].max()),3)
        universe_size = scipy.special.binom(matrix_size, region_size - start_diagonal + 1)
        num_unique_samples = compute_expected_unique_samples(total_items=universe_size,
                                                             num_samples=len(shuffled_samples[region_size]))
        this_fit_params = distribution_class.fit(shuffled_samples[region_size],
                                                 support_range=support_ranges[region_size],
                                                 max_pvalue_cv=max_pvalue_std_error, **fit_kwargs)
        fit_params[region_size] = this_fit_params
        log_print('region size: {}, fit parameters: {}'.format(region_size, this_fit_params), 3)

    return smooth_parameters(fit_params, parameter_smoothing_window_size=parameter_smoothing_window_size)


def compute_expected_unique_samples(total_items, num_samples):
    """
    Return the expected number of unique items from a set of size :param total_items:
    in a sample of size :param num_samples: with replacement.
    """
    unsampled_items = scipy.stats.binom(num_samples, 1/total_items).pmf(0)
    if unsampled_items < 1.0:
        return int((1 - unsampled_items) * total_items)
    else:
        return num_samples

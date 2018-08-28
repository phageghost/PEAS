import pandas
from scipy.signal import savgol_filter

from peas.utilities import log_print, force_odd

SAVGOL_DEFAULT_WINDOW_SIZE = 5


def smooth_parameters(param_dict, filter_window_size=SAVGOL_DEFAULT_WINDOW_SIZE):
    # ToDo: Refactor for elegance
    if len(param_dict) >= 3:
        if filter_window_size:
            filter_window_size = max(force_odd(int(filter_window_size)), 3)
            log_print('Smoothing parameters with Savitsky-Golay filter of size {}'.format(filter_window_size), 3)
            param_df = pandas.DataFrame(param_dict).T  # ToDo: refactor to remove pandas dependency here.
            param_array = savgol_filter(param_df, filter_window_size, 1, axis=0)
            param_dict = {region_size: params for region_size, params in
                          zip(sorted(param_dict.keys()), param_array.tolist())}
    else:
        log_print('Too few region sizes to perform parameter smoothing (need at least 3)', 3)

    return param_dict


def fit_distros(shuffled_samples, distribution_class, filter_window_size=SAVGOL_DEFAULT_WINDOW_SIZE, fit_kwargs={}):
    """
    Given a dictionary of permuted data vectors, return a dictionary of optimal parameters
    (as tuples) for distributions of class :param:`distro_class`.
    """
    sizes = sorted(shuffled_samples.keys())

    fit_params = {}
    for region_size in sizes:
        # log_print('size {}, min score: {}, mean score: {}, max score: {}'.format(region_size, sampled_scores[region_size].min(), sampled_scores[region_size].mean(), sampled_scores[region_size].max()),3)
        this_fit_params = distribution_class.fit(shuffled_samples[region_size], **fit_kwargs)
        fit_params[region_size] = this_fit_params
        log_print('size: {} fit parameters: {}'.format(region_size, this_fit_params), 3)

    return smooth_parameters(fit_params, filter_window_size=filter_window_size)

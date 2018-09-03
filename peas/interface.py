import empdist
import numpy
import scipy.stats

from peas.arrayfuncs import replace_nans_diagonal_means, compute_vector_trim_points, compute_matrix_trim_points, \
    create_diagonal_distance_matrix, create_data_masks
from peas.fitapproxdistros import distributions
from peas.utilities import log_print, gaussian_norm, validate_param
from . import choosing
from . import region_stats
from . import scoring

DEFAULT_PVALUE_TARGET = 1e-6
MAX_PSCORE = 744.44007192138122
MIN_PVALUE = numpy.exp(-MAX_PSCORE)
MAX_PVAL = 1 - 1e-100
DEFAULT_PVALUE_CV = 0.05

SCORING_FUNCS_BY_NAME = {'sum': scoring.compute_sum_table_2d,
                         'mean': scoring.compute_mean_table_2d,
                         'min': scoring.compute_min_table_2d,
                         'max': scoring.compute_max_table_2d}

NULL_DISTRIBUTIONS_BY_NAME = {'pw_power': distributions.PiecewiseApproxPower,
                              'pw_linear': distributions.PiecewiseApproxLinear}

DEFAULT_PARAMETER_SMOOTHING_METHOD = 'savgol'
SAVGOL_DEFAULT_WINDOW_SIZE = 5
DEFAULT_NULL_DISTRIBUTION, = 'pw_power'


def find_ropes(input_data, score_method='mean', min_score=0, max_pval=None, min_size=2, max_size=None,
               trim_input=True, trim_edges=False, gobig=True, tail=None,
               pvalue_target=DEFAULT_PVALUE_TARGET, start_diagonal=1,
               quantile_normalize=False, more_smoothing=False,
               edge_weight_constant=0, edge_weight_power=1,
               return_debug_data=False, parameter_filter_strength=0, random_seed=None):
    assert len(input_data.shape <= 2), 'Input array has too many dimensions (max 2).'

    if len(input_data.shape) == 1:
        log_print('Input is 1D vector')
        find_ropes_vector()

    elif len(input_data.shape) == 2:
        log_print('Input is 2D matrix')
        find_ropes_matrix()


def find_ropes_vector(input_vector, min_score=0, max_pval=None, min_size=2, max_size=None,
                                 trim_input=True, trim_edges=False, gobig=True, tail=None,
                                 pvalue_target=DEFAULT_PVALUE_TARGET, start_diagonal=1,
                                 quantile_normalize=False, more_smoothing=False,
                                 edge_weight_constant=0, edge_weight_power=1,
                                 return_debug_data=False, parameter_filter_strength=0, random_seed=None):
    numpy.random.seed(random_seed)
    input_vector = trim_data_vector(input_vector)
    if quantile_normalize:
        log_print('quantile-normalizing vector to standard Gaussian ...', 2)
        input_vector = gaussian_norm(input_vector)

    region_scores, empirical_distro = generate_score_distributions_vector(input_vector=input_vector)
    pscores = region_stats.compute_pscores(region_scores=region_scores, empirical_distros=empirical_distro)


def find_ropes_matrix(input_matrix, min_score=0, max_pval=None, min_size=2, max_size=None,
                      tail='both',
                      maximization_target='p_prod',
                      edge_weight_power=1,
                      start_diagonal=1,
                      quantile_normalize=False,
                      parameter_filter_strength=0,
                      random_seed=None,
                      gobig=True, ):
    numpy.random.seed(random_seed)
    assert input_matrix.shape[0] == input_matrix.shape[1], 'Input matrix must be square.'

    input_matrix = trim_data_matrix(input_matrix)
    n = input_matrix.shape[0]

    if not max_size:
        max_size = n // 2
    else:
        max_size = min(max_size, n)
    assert start_diagonal < min_size

    input_matrix = replace_nans_diagonal_means(input_matrix, start_diagonal=start_diagonal,
                                               end_diagonal=max_size)  # ToDo: Handle unsquare trimming results

    if quantile_normalize:
        log_print('quantile-normalizing matrix to standard Gaussian ...', 2)
        input_matrix = gaussian_norm(input_matrix.flatten()).reshape((n, n))

    region_scores, empirical_distro = generate_score_distributions_matrix(input_matrix=input_matrix)
    pval_scores = region_stats.compute_pscores(region_scores=region_scores, empirical_distros=empirical_distro,
                                               tail=tail)
    pvals = region_stats.convert_pscores_to_pvals(pscores=pval_scores)
    row_masks, col_masks = generate_region_masks(pval_scores=pval_scores, min_size=min_size, max_size=max_size,
                                                 min_score=min_score, max_pval=max_pval)
    edge_weights = compute_edge_weights(region_scores, region_pvals=pvals, pval_scores=pval_scores,
                                        empirical_distributions=empirical_distro, min_size=min_size, max_size=max_size,
                                        maximization_target=maximization_target, edge_weight_power=edge_weight_power)

    regions = choosing.pick_regions(edge_weights=edge_weights, row_masks=row_masks, col_masks=col_masks, gobig=gobig)


def trim_data_vector(input_vector):
    trim_start, trim_end = compute_vector_trim_points((input_vector))
    trimmed_vector = input_vector[trim_start:trim_end]
    log_print(
        'trimmed {} element vector to remove preceding and trailing NaNs. {} elements remain'.format(len(input_vector),
                                                                                                     len(
                                                                                                         trimmed_vector)))
    return trimmed_vector


def trim_data_matrix(input_matrix):
    row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point = compute_matrix_trim_points(
        input_matrix)
    trimmed_matrix = input_matrix[row_start_trim_point:row_end_trim_point, col_start_trim_point:col_end_trim_point]
    log_print('trimmed {} x {} matrix to remove contiguous NaNs, now {} x {}.'.format(*input_matrix.shape,
                                                                                      *trimmed_matrix.shape))
    return trimmed_matrix


# def generate_score_distributions_vector(input_vector, min_score, max_pval, min_size, max_size,
#                                         start_diagonal=1,
#                                         gobig=True, tail=None,
#                                         pvalue_target=DEFAULT_PVALUE_TARGET,
#                                         quantile_normalize=False, more_smoothing=False,
#                                         edge_weight_constant=0, edge_weight_power=1,
#                                         return_debug_data=False, parameter_filter_strength=0, random_seed=None):
#
#     n = len(input_vector)
#     max_distro_size = max_size + 1  # ToDo: clean up
#
#     log_print('computing means of all subarrays of {}-element vector ...'.format(n), 2)
#     region_scores = scoring.compute_mean_table_1d(input_vector)
#     # flat_cell_values = input_data
#     if not tail: tail = 'both'
#     log_print('constructing null models for regions up to size {} ...'.format(max_size), 2)
#     null_data = region_stats.generate_permuted_matrix_scores()
#     empirical_distros = distributions.generate_empirical_distributions_region_means(data=input_vector,
#                                                                                     max_region_size=max_distro_size,
#                                                                                     num_bins=EMPIRICAL_BINS,
#                                                                                     max_empirical_size=max_distro_size,
#                                                                                     support=(input_vector.min(),
#                                                                                              input_vector.max())
#                                                                                     )
#
#
#     return region_scores, empirical_distros


def generate_score_distributions_matrix(input_matrix,
                                        min_size, max_size,
                                        score_func='mean',
                                        start_diagonal=1,
                                        tail='both',
                                        pvalue_target=DEFAULT_PVALUE_TARGET,
                                        max_pvalue_cv=DEFAULT_PVALUE_CV,
                                        parameter_smoothing_method=DEFAULT_PARAMETER_SMOOTHING_METHOD,
                                        parameter_filter_strength=SAVGOL_DEFAULT_WINDOW_SIZE,
                                        num_shuffles='auto',
                                        null_distribution_class=DEFAULT_NULL_DISTRIBUTION,
                                        random_seed=None):

    assert input_matrix.shape[0] == input_matrix.shape[1], 'Input matrix must be square.'
    validate_param('score_func', score_func, SCORING_FUNCS_BY_NAME.keys())
    validate_param('null_distribution_class', null_distribution_class, NULL_DISTRIBUTIONS_BY_NAME.keys())
    n = input_matrix.shape[0]
    # if not tail: tail = 'right' # ToDo: Adapt code to fit logsf to allow 'left' and 'both' values.
    assert tail == 'right'

    log_print('computing means of all diagonal square subsets of {} x {} matrix ...'.format(n, n), 2)
    region_scores = scoring.compute_mean_table_2d(input_matrix, start_diagonal=start_diagonal)

    # Automatic determination of number of shuffles needed to achieve p-value target based on region sizes.
    if num_shuffles == 'auto':
        num_shuffles = empdist.empirical_pval.compute_number_of_permuted_data_points(target_p_value=pvalue_target,
                                                                                     max_pvalue_cv=max_pvalue_cv)

    log_print(
        'constructing null models for regions up to size {} using {} permutations ...'.format(max_size,
                                                                                              num_shuffles), 2)

    shuffled_samples = region_stats.generate_permuted_matrix_scores(matrix=input_matrix,
                                                                    num_shuffles=num_shuffles,
                                                                    min_region_size=min_size,
                                                                    max_region_size=max_size,
                                                                    start_diagonal=start_diagonal,
                                                                    matrix_score_func=SCORING_FUNCS_BY_NAME[score_func],
                                                                    random_seed=random_seed)

    null_distributions = region_stats.fit_distributions(sampled_scores=shuffled_samples,
                                                        distribution_class=NULL_DISTRIBUTIONS_BY_NAME[
                                                            null_distribution_class],
                                                        parameter_smoothing_method=parameter_smoothing_method,
                                                        parameter_smoothing_window_size=parameter_filter_strength)

    return region_scores, null_distributions


def generate_region_masks(region_scores, pval_scores, min_size, max_size, min_score=0, max_pval=None):
    assert region_scores.shape[0] == region_scores.shape[1]
    assert pval_scores.shape[0] == pval_scores.shape[1]
    assert region_scores.shape[0] == pval_scores.shape[0]
    n = region_scores.shape[0]
    # Apply filters to generate masks

    log_print('applying filters ...', 2)

    mask_2d = numpy.zeros((n, n), dtype=bool)

    log_print('minimum size: {} ...'.format(min_size), 3)
    mask_2d[numpy.triu_indices(n, min_size - 1)] = True

    if max_size < n:
        log_print('maximum size: {} ...'.format(max_size), 3)
        mask_2d[numpy.triu_indices(n, max_size)] = False

    if min_score > 0:
        log_print('minimum absolute score: {} ...'.format(min_score), 3)
        mask_2d = numpy.logical_and(mask_2d, numpy.greater(numpy.abs(region_scores), min_score))

    if max_pval is not None:
        log_print('maximum p-value: {} ...'.format(max_pval), 3)
        p_score_threshold = -numpy.log(max_pval)
        mask_2d = numpy.logical_and(mask_2d, numpy.greater(pval_scores, p_score_threshold))

    row_masks, col_masks = create_data_masks(mask_2d)
    return row_masks, col_masks


def compute_edge_weights(region_scores, region_pvals, pval_scores, empirical_distributions,
                         min_size, max_size, maximization_target='p_prod', edge_weight_power=2):
    validate_param('maximization_target', maximization_target,
                          ('p_prod', 'coverage', 'score', 'information', 'z'))
    assert region_scores.shape[0] == region_scores.shape[1]
    assert region_pvals.shape[0] == region_pvals.shape[1]
    assert pval_scores.shape[0] == pval_scores.shape[1]
    assert region_scores.shape[0] == region_pvals.shape[0]
    assert region_scores.shape[0] == pval_scores.shape[0]
    n = pval_scores.shape[0]


    if maximization_target == 'p_prod':
        log_print('minimizing product of p-values', 2)
        edge_weights = pval_scores.copy()

    elif maximization_target == 'coverage':
        log_print('maximizing coverage', 2)
        edge_weights = create_diagonal_distance_matrix(n).astype(float)

    elif maximization_target == 'score':  # with mean scores this will just pick up minimum size regions so is pretty useless at the moment.
        log_print('maximizing combined score', 2)
        edge_weights = region_scores.copy()

    elif maximization_target == 'information':
        log_print('maximizing information content', 2)
        edge_weights = region_stats.compute_information_matrix(region_scores, empirical_distributions,
                                                               diagonal_start=min_size - 1, diagonal_end=max_size)

    elif maximization_target == 'z':
        log_print('maximizing standard z score of p-values', 2)
        edge_weights = region_pvals.copy()
        edge_weights[numpy.equal(region_pvals, 1)] = MAX_PVAL 

        edge_weights[numpy.triu_indices(n, 1)] = -scipy.stats.norm().ppf(edge_weights[numpy.triu_indices(n, 1)])

    if edge_weight_power != 1:
        log_print('raising edge weights to power of {}'.format(edge_weight_power), 2)
        edge_weights **= edge_weight_power

    return edge_weights




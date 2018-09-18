import empdist
import numpy
import scipy.stats

from peas.arrayfuncs import replace_nans_diagonal_means, compute_vector_trim_points, compute_matrix_trim_points, \
    create_diagonal_distance_matrix, create_data_masks
from peas.utilities import log_print, gaussian_norm, validate_param
from . import choosing
from . import constants
from . import region_stats
from . import scoring


# def find_peas(input_data, score_method='mean', min_score=0, max_pval=None, min_size=2, max_size=None,
#               trim_input=True, trim_edges=False, gobig=True, tail=None,
#               pvalue_target=constants.DEFAULT_PVALUE_TARGET, start_diagonal=1,
#               quantile_normalize=False, more_smoothing=False,
#               edge_weight_constant=0, edge_weight_power=1,
#               return_debug_data=False, parameter_filter_strength=0, random_seed=None):
#     assert len(input_data.shape <= 2), 'Input array has too many dimensions (max 2).'
#
#     if len(input_data.shape) == 1:
#         log_print('Input is 1D vector')
#         find_peas_vector()
#
#     elif len(input_data.shape) == 2:
#         log_print('Input is 2D matrix')
#         find_peas_matrix()

# ToDo: Add command-line script to process non-genomic CSV files.

def find_peas_vector(input_vector, min_score=constants.DEFAULT_MIN_SCORE, max_pval=constants.DEFAULT_PVALUE_THRESHOLD,
                     min_size=constants.DEFAULT_MIN_SIZE, max_size=constants.DEFAULT_MAX_SIZE,
                     maximization_target=constants.DEFAULT_MAXIMIZATION_TARGET,
                     tail=constants.DEFAULT_TAIL,
                     bins=constants.DEFAULT_BINS,
                     quantile_normalize=False,
                     edge_weight_power=constants.DEFAULT_ALPHA,
                     gobig=True):
    input_vector, trim_start, trim_end = trim_data_vector(
        input_vector)  # ToDo: Move this upstream so we can decide on whether to process based on trimming results.
    n = len(input_vector)
    assert n > min_size + 1
    # ToDo: move data normalization steps from genomic_regions to here so they can operate NaN-free

    if quantile_normalize:
        log_print('quantile-normalizing vector to standard Gaussian ...', 2)
        input_vector = gaussian_norm(input_vector)

    half_vector_length = n // 2

    if not max_size:
        max_size = max(n // constants.DEFAULT_MAX_SIZE_FACTOR, min_size + 1)
    elif max_size > half_vector_length:
        log_print('specified max size of {} is too large, setting to half vector length {}'.format(max_size,
                                                                                                   half_vector_length))
        max_size = half_vector_length

    assert max_size >= min_size

    region_scores, null_distributions = generate_score_distributions_vector(input_vector=input_vector,
                                                                            min_size=min_size, max_size=max_size,
                                                                            bins=bins,
                                                                            score_func_name='mean',
                                                                            pseudocount=constants.DEFAULT_PSEUDOCOUNT)

    return find_peas_common(region_scores=region_scores,
                            null_distributions=null_distributions,
                            trim_start=trim_start,
                            tail=tail, maximization_target=maximization_target,
                            min_size=min_size, max_size=max_size,
                            min_score=min_score, max_pval=max_pval,
                            edge_weight_power=edge_weight_power,
                            gobig=gobig)


def find_peas_matrix(input_matrix,
                     score_func_name=constants.DEFAULT_SCORE_FUNC,
                     min_score=0, max_pval=None, min_size=2, max_size=None,
                     tail='both',
                     pvalue_target=constants.DEFAULT_PVALUE_TARGET,
                     max_pvalue_cv=constants.DEFAULT_PVALUE_CV,
                     num_shuffles='auto',
                     null_distribution_class=constants.DEFAULT_NULL_DISTRIBUTION,
                     maximization_target=constants.DEFAULT_MAXIMIZATION_TARGET,
                     edge_weight_power=1,
                     start_diagonal=1,
                     quantile_normalize=False,
                     parameter_smoothing_method=constants.DEFAULT_PARAMETER_SMOOTHING_METHOD,
                     parameter_filter_strength=constants.DEFAULT_PARAMETER_SMOOTHING_WINDOW_SIZE,
                     random_seed=None,
                     gobig=True):
    assert input_matrix.shape[0] == input_matrix.shape[1], 'input matrix must be square.'

    trimmed_matrix, row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point = trim_data_matrix(
        input_matrix)
    n = trimmed_matrix.shape[0]  # ToDo: Gracefully handle case where trimmed data is no longer big enough to work with
    assert n > min_size + 1

    half_matrix_size = n // 2

    if not max_size:
        max_size = max(n // constants.DEFAULT_MAX_SIZE_FACTOR, min_size + 1)
    elif max_size > half_matrix_size:
        log_print(
            'specified max size of {} is too large for trimmed matrix of size {} x {}, setting to half matrix size {}'.format(
                max_size, n, n, half_matrix_size))
        max_size = half_matrix_size

    assert max_size >= min_size
    assert start_diagonal < min_size

    trimmed_matrix = replace_nans_diagonal_means(trimmed_matrix, start_diagonal=start_diagonal,
                                                 end_diagonal=max_size - 1)  # ToDo: Handle unsquare trimming results

    if quantile_normalize:
        log_print('quantile-normalizing matrix to standard Gaussian ...', 2)
        trimmed_matrix = gaussian_norm(trimmed_matrix.flatten()).reshape((n, n))

    region_scores, null_distributions = generate_score_distributions_matrix(input_matrix=trimmed_matrix,
                                                                            min_size=min_size, max_size=max_size,
                                                                            score_func_name=score_func_name,
                                                                            start_diagonal=start_diagonal,
                                                                            tail=tail,
                                                                            pvalue_target=pvalue_target,
                                                                            max_pvalue_cv=max_pvalue_cv,
                                                                            parameter_smoothing_method=parameter_smoothing_method,
                                                                            parameter_filter_strength=parameter_filter_strength,
                                                                            num_shuffles=num_shuffles,
                                                                            null_distribution_class=null_distribution_class,
                                                                            random_seed=random_seed)
    return find_peas_common(region_scores=region_scores,
                            null_distributions=null_distributions,
                            trim_start=row_start_trim_point,
                            tail=tail, maximization_target=maximization_target,
                            min_size=min_size, max_size=max_size,
                            min_score=min_score,
                            max_pval=max_pval,
                            edge_weight_power=edge_weight_power,
                            gobig=gobig)


def find_peas_common(region_scores, null_distributions, trim_start, tail, maximization_target, min_score, min_size,
                     max_size, max_pval, edge_weight_power, gobig):
    pscores = region_stats.compute_pscores(region_scores=region_scores, null_distributions=null_distributions,
                                           tail=tail)
    pvals = region_stats.convert_pscores_to_pvals(pscores=pscores)
    row_masks, col_masks = generate_region_masks(region_scores=region_scores, pval_scores=pscores, min_size=min_size,
                                                 max_size=max_size,
                                                 min_score=min_score, max_pval=max_pval)
    edge_weights = compute_edge_weights(region_scores, region_pvals=pvals, pval_scores=pscores,
                                        empirical_distributions=null_distributions, min_size=min_size,
                                        max_size=max_size,
                                        maximization_target=maximization_target, edge_weight_power=edge_weight_power)
    region_coords = choosing.pick_regions(edge_weights=edge_weights, row_masks=row_masks, col_masks=col_masks,
                                          gobig=gobig)
    regions = [(start + trim_start, end + trim_start, end - start + 1, region_scores[start, end],
                pvals[start, end]) for start, end in region_coords[::-1]]

    return regions


def trim_data_vector(input_vector):
    trim_start, trim_end = compute_vector_trim_points(input_vector)
    trimmed_vector = input_vector[trim_start:trim_end]
    log_print(
        'trimmed {} element vector to remove preceding and trailing NaNs. {} elements remain'.format(len(input_vector),
                                                                                                     len(
                                                                                                         trimmed_vector)),
        2)
    return trimmed_vector, trim_start, trim_end


def trim_data_matrix(input_matrix):
    row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point = compute_matrix_trim_points(
        input_matrix)
    trimmed_matrix = input_matrix[row_start_trim_point:row_end_trim_point, col_start_trim_point:col_end_trim_point]
    log_print('trimmed {} x {} matrix to remove contiguous NaNs, now {} x {}.'.format(*input_matrix.shape,
                                                                                      *trimmed_matrix.shape), 2)
    return trimmed_matrix, row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point


def generate_score_distributions_vector(input_vector, min_size, max_size,
                                        bins=constants.DEFAULT_BINS,
                                        score_func_name='mean',
                                        pseudocount=constants.DEFAULT_PSEUDOCOUNT):
    validate_param('score_func_name', score_func_name, constants.VECTOR_SCORE_FUNCS_BY_NAME.keys())
    n = len(input_vector)

    log_print('computing means of all subarrays of {}-element vector ...'.format(n), 2)
    region_scores = scoring.compute_mean_table_1d(input_vector, end_diagonal=max_size - 1)

    log_print('constructing null models for regions up to size {} ...'.format(max_size), 2)

    singleton_distribution = empdist.EmpiricalDistribution.from_data(data=input_vector,
                                                                     bins=bins,
                                                                     pseudocount=pseudocount)  # type: empdist.EmpiricalDistribution

    score_func = constants.VECTOR_SCORE_FUNCS_BY_NAME[score_func_name]  # type: function

    null_distributions = score_func(input_empirical_distribution=singleton_distribution, max_sample_size=max_size)
    null_distributions = {i: null_distributions[i] for i in range(min_size, max_size + 1)}

    return region_scores, null_distributions


def generate_score_distributions_matrix(input_matrix,
                                        min_size, max_size,
                                        score_func_name='mean',
                                        start_diagonal=1,
                                        tail='both',
                                        pvalue_target=constants.DEFAULT_PVALUE_TARGET,
                                        max_pvalue_cv=constants.DEFAULT_PVALUE_CV,
                                        parameter_smoothing_method=constants.DEFAULT_PARAMETER_SMOOTHING_METHOD,
                                        parameter_filter_strength=constants.DEFAULT_PARAMETER_SMOOTHING_WINDOW_SIZE,
                                        num_shuffles='auto',
                                        null_distribution_class=constants.DEFAULT_NULL_DISTRIBUTION,
                                        random_seed=None):
    assert input_matrix.shape[0] == input_matrix.shape[1], 'Input matrix must be square.'
    validate_param('score_func', score_func_name, constants.MATRIX_SCORING_FUNCS_BY_NAME.keys())
    validate_param('null_distribution_class', null_distribution_class, constants.NULL_DISTRIBUTIONS_BY_NAME.keys())
    n = input_matrix.shape[0]
    # if not tail: tail = 'right' # ToDo: Adapt code to fit logsf to allow 'left' and 'both' values.
    assert tail == 'right'

    log_print('computing means of all diagonal square subsets of {} x {} matrix ...'.format(n, n), 2)
    region_scores = scoring.compute_mean_table_2d(input_matrix, start_diagonal=start_diagonal)

    support_ranges = {}
    for diag_idx in range(start_diagonal, max_size):
        this_scores = numpy.diag(region_scores, diag_idx)
        support_ranges[diag_idx + 1] = (this_scores.min(), this_scores.max())

    # Automatic determination of number of shuffles needed to achieve p-value target based on region sizes.
    if num_shuffles == 'auto':
        num_shuffles = empdist.empirical_pval.compute_number_of_permuted_data_points(target_p_value=pvalue_target,
                                                                                     max_pvalue_cv=max_pvalue_cv) // (
                               n - (max_size - 1))

    log_print(
        'constructing null models for regions up to size {} using {} permutations ...'.format(max_size,
                                                                                              num_shuffles), 2)

    shuffled_samples = region_stats.generate_permuted_matrix_scores(matrix=input_matrix,
                                                                    num_shuffles=num_shuffles,
                                                                    min_region_size=min_size,
                                                                    max_region_size=max_size,
                                                                    start_diagonal=start_diagonal,
                                                                    matrix_score_func=
                                                                    constants.MATRIX_SCORING_FUNCS_BY_NAME[
                                                                        score_func_name],
                                                                    random_seed=random_seed)

    null_distributions = region_stats.fit_distributions(sampled_scores=shuffled_samples,
                                                        matrix_size=n,
                                                        start_diagonal=start_diagonal,
                                                        distribution_class=constants.NULL_DISTRIBUTIONS_BY_NAME[
                                                            null_distribution_class],
                                                        support_ranges=support_ranges,
                                                        parameter_smoothing_method=parameter_smoothing_method,
                                                        parameter_smoothing_window_size=parameter_filter_strength)

    return region_scores, null_distributions


def generate_region_masks(region_scores, pval_scores, min_size, max_size, min_score=0, max_pval=None):
    assert region_scores.shape[0] == region_scores.shape[1]
    assert pval_scores.shape[0] == pval_scores.shape[1]
    assert region_scores.shape[0] == pval_scores.shape[0]
    n = region_scores.shape[0]
    # Apply filters to generate masks

    log_print('applying filters', 2)

    mask_2d = numpy.zeros((n, n), dtype=bool)

    log_print('minimum size: {}'.format(min_size), 3)
    mask_2d[numpy.triu_indices(n, min_size - 1)] = True

    if max_size < n:
        log_print('maximum size: {}'.format(max_size), 3)
        mask_2d[numpy.triu_indices(n, max_size)] = False

    if min_score > 0:
        log_print('minimum absolute score: {}'.format(min_score), 3)
        mask_2d = numpy.logical_and(mask_2d, numpy.greater(numpy.abs(region_scores), min_score))

    if max_pval is not None:
        log_print('maximum p-value: {}'.format(max_pval), 3)
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

    if maximization_target == 'coverage':
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
        edge_weights[numpy.equal(region_pvals, 1)] = constants.MAX_PVAL

        edge_weights[numpy.triu_indices(n, 1)] = -scipy.stats.norm().ppf(edge_weights[numpy.triu_indices(n, 1)])

    else:
        log_print('minimizing product of p-values', 2)
        edge_weights = pval_scores.copy()

    if edge_weight_power != 1:
        log_print('raising edge weights to power of {}'.format(edge_weight_power), 2)
        edge_weights **= edge_weight_power

    return edge_weights

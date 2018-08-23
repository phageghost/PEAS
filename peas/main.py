from .utilities import *

DEFAULT_PVALUE_TARGET = 1e-6


def generate_score_distributions(min_score=0, max_pval=None, min_size=2, max_size=None,
                                 trim_input=True, trim_edges=False, gobig=True, tail=None,
                                 pvalue_target=DEFAULT_PVALUE_TARGET, start_diagonal=1,
                                 quantile_normalize=False, more_smoothing=False,
                                 edge_weight_constant=0, edge_weight_power=1,
                                 return_debug_data=False, parameter_filter_strength=0, random_seed=None):
    """
    Given a peak correlation matrix, return a list of tuples containing information
    about an optimal set regions such that the number of elements covered by regions with a mean
    value of at least :param:`score_threshold` and a size of at least :param:`min_size` is maximized.

    If :param:`trim-edges` is True, remove points from these regions whose corresponding
    rows fall below :param:`score_threshold`.
    """
    original_n = input_data.shape[0]

    input_data = chrom_data[chrom]  # Get from global inherited data instead of passing as a parameter

    if len(input_data.shape) == 2 and trim_input:
        log_print('{}: trimming {} x {} matrix to remove contiguous NaNs ...'.format(chrom, *input_data.shape), 2)
        row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point = compute_matrix_trim_points(
            input_data)
        input_data = input_data[row_start_trim_point:row_end_trim_point, col_start_trim_point:col_end_trim_point]
        log_print('{}: trimmed matrix is now {} x {}'.format(chrom, *input_data.shape), 3)

        input_data = replace_nans_diagonal_means(input_data, start_diagonal=start_diagonal,
                                                 end_diagonal=0)  # ToDo: Handle unsquare trimming results


    else:  # ToDo: perform equivalent trimming for 1D vectors
        row_start_trim_point = 0
    if trim_edges:
        assert min_score > 0  # since edge trimming works on scores of rows, we need a score threshold not just a p-value threshold.
    if len(input_data.shape) > 1:
        assert input_data.shape[0] == input_data.shape[1]
    n = input_data.shape[0]
    if not max_size:
        max_size = n // 2
    else:
        max_size = min(max_size, n)
    assert start_diagonal < min_size
    max_distro_size = max_size + 1  # ToDo: clean up

    if quantile_normalize:
        log_print('{}: quantile-normalizing matrix to standard Gaussian ...'.format(chrom), 2)
        input_data = gaussian_norm(input_data.flatten()).reshape((n, n))

    if len(input_data.shape) == 1:
        log_print('{}: computing means of all subarrays of {}-element vector ...'.format(chrom, n), 2)
        region_scores = simstraps.compute_mean_table_1d(input_data)
        # flat_cell_values = input_data
        if not tail: tail = 'both'
        log_print('{}: constructing null models for regions up to size {} ...'.format(chrom, max_size), 2)
        empirical_distros = simstraps.generate_empirical_distributions_region_means(data=input_data,
                                                                                    max_region_size=max_distro_size,
                                                                                    num_bins=EMPIRICAL_BINS,
                                                                                    max_empirical_size=max_distro_size,
                                                                                    support=(input_data.min(),
                                                                                             input_data.max())
                                                                                    )


    else:
        log_print('{}: computing means of all diagonal square subsets of {} x {} matrix ...'.format(chrom, n, n), 2)
        region_scores = simstraps.compute_mean_table_2d(input_data, start_diagonal=start_diagonal)
        if not tail: tail = 'right'

        # Automatic determination of number of shuffles needed to achieve p-value target based on region sizes. # ToDo: make this better / more principled. Possibly iterative.
        num_shuffles = int(numpy.ceil((1 / pvalue_target / (n - max_size + 1))))

        log_print(
            '{}: constructing null models for regions up to size {} using {} permutations ...'.format(chrom, max_size,
                                                                                                      num_shuffles), 2)

        empirical_distros = simstraps.generate_empirical_distributions_dependent_region_means(matrix=input_data,
                                                                                              num_shuffles=num_shuffles,
                                                                                              min_region_size=min_size,
                                                                                              max_region_size=max_distro_size,
                                                                                              start_diagonal=start_diagonal,
                                                                                              random_seed=random_seed,
                                                                                              filter_window_size=parameter_filter_strength)

    if max_pval is not None:
        log_print('{}: computing -log pvalues ...'.format(chrom), 2)

        if len(input_data.shape) == 1:  # ToDo: Do logsf with flexible tails to avoid this hackiness
            region_pvals = simstraps.compute_pvalues_matrix(data_matrix=region_scores,
                                                            distro_dict=empirical_distros,
                                                            diagonal_start=min_size - 1,
                                                            diagonal_end=max_distro_size,
                                                            tail=tail)
            pval_scores = -numpy.log(numpy.maximum(region_pvals, MIN_PVALUE))
        #             if not return_debug_data:
        #                 del(region_pvals)
        else:
            pval_scores = simstraps.compute_pscores_matrix(data_matrix=region_scores,
                                                           distro_dict=empirical_distros,
                                                           diagonal_start=min_size - 1,
                                                           diagonal_end=max_distro_size,
                                                           tail=tail)

            # log_print('{}: bounding -log pvalues at {} ...'.format(chrom, MIN_PSCORE), 2)
            # print(pval_scores.min(), pval_scores.max())
            # pval_scores = numpy.minimum(pval_scores, MIN_PSCORE)
            region_pvals = numpy.exp(-numpy.minimum(pval_scores, MAX_PSCORE))

    # Apply filters to generate masks

    log_print('{}: applying filters ...'.format(chrom), 2)

    mask_2d = numpy.zeros((n, n), dtype=bool)

    log_print('{}: minimum size: {} ...'.format(chrom, min_size), 3)
    mask_2d[numpy.triu_indices(n, min_size - 1)] = True

    if max_size < n:
        log_print('{}: maximum size: {} ...'.format(chrom, max_size), 3)
        mask_2d[numpy.triu_indices(n, max_size)] = False

    if min_score > 0:
        log_print('{}: minimum absolute score: {} ...'.format(chrom, min_score), 3)
        mask_2d = numpy.logical_and(mask_2d, numpy.greater(numpy.abs(region_scores), min_score))

    if max_pval is not None:
        log_print('{}: maximum p-value: {} ...'.format(chrom, max_pval), 3)
        p_score_threshold = -numpy.log(max_pval)
        mask_2d = numpy.logical_and(mask_2d, numpy.greater(pval_scores, p_score_threshold))

    row_masks, col_masks = create_data_masks(mask_2d)

    # Optimize region sets for given maximization target

    if maximization_target == 'p_prod':
        log_print('{}: minimizing product of p-values'.format(chrom), 2)
        edge_weights = pval_scores.copy()

    elif maximization_target == 'p_sum':
        log_print('{}: minimizing sum of p-values'.format(chrom), 2)
        edge_weights = -region_pvals.copy()

    elif maximization_target == 'coverage':
        log_print('{}: maximizing coverage'.format(chrom), 2)
        edge_weights = create_diagonal_distance_matrix(n).astype(float)

    elif maximization_target == 'score':  # with mean scores this will just pick up minimum size regions so is pretty useless at the moment.
        log_print('{}: maximizing combined mean'.format(chrom), 2)
        edge_weights = region_scores.copy()

    elif maximization_target == 'information':
        log_print('{}: maximizing information content'.format(chrom), 2)
        edge_weights = simstraps.compute_information_matrix(region_scores, empirical_distros,
                                                            diagonal_start=min_size - 1, diagonal_end=max_size)
        print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())

    elif maximization_target == 'z':
        log_print('{}: maximizing standard z score of p-values'.format(chrom), 2)
        edge_weights = region_pvals.copy()
        # print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())
        edge_weights[numpy.equal(region_pvals, 1)] = MAX_PVALUE
        # print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())
        # print(edge_weights)

        edge_weights[numpy.triu_indices(n, 1)] = -scipy.stats.norm().ppf(edge_weights[numpy.triu_indices(n, 1)])
        # print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())

    elif maximization_target == 'wz':
        log_print('{}: maximizing coverage-weighted z score of p-values'.format(chrom), 2)
        edge_weights = region_pvals.copy()
        edge_weights[numpy.equal(region_pvals, 1)] = MAX_PVALUE
        edge_weights[numpy.triu_indices(n, 1)] = -scipy.stats.norm().ppf(edge_weights[numpy.triu_indices(n, 1)])
        edge_weights *= create_diagonal_distance_matrix(n).astype(float)

    if more_smoothing:
        log_print('{}: performing additional smoothing of edge weights'.format(chrom), 2)
        edge_weights = simstraps.compute_mean_table_2d(edge_weights, start_diagonal=min_size - 1, end_diagonal=max_size)

    if edge_weight_constant != 0:
        # print(edge_weights.min(), edge_weights.mean(), edge_weights.max())
        log_print('{}: adding {} to all edge weights'.format(chrom, edge_weight_constant), 2)

        edge_weights += edge_weight_constant
        # print(edge_weights.min(), edge_weights.mean(), edge_weights.max())

    if edge_weight_power != 1:
        log_print('{}: raising edge weights to power of {}'.format(chrom, edge_weight_power), 2)
        edge_weights **= edge_weight_power

    #     if not return_debug_data:
    #         del(input_data)
    #         del(mask_2d)
    #         del(pval_scores)

    log_print('{}: computing optimum regions ...'.format(chrom), 2)
    if return_debug_data:
        original_edge_weights = edge_weights.copy()
    score_vec, backtrack = find_maximal_path(edge_weights, row_masks, col_masks, gobig=gobig)
    debug_print(score_vec, backtrack)

    region_coords = decode_backtrack(backtrack)

    if trim_edges:
        log_print('{}: refining region edges ...'.format(chrom), 2)
        region_coords = trim_regions(data=correlation_matrix,
                                     region_coords=region_coords,
                                     min_size=min_size,
                                     edge_trim_threshold=min_score)

    regions = [(start + row_start_trim_point, end + row_start_trim_point, end - start + 1, region_scores[start, end],
                region_pvals[start, end]) for start, end in region_coords[::-1]]

    if return_debug_data:
        if len(input_data.shape) == 2 and trim_input:
            # Reconstruct matrices that correspond to the untrimmed input matrix
            def reconstruct_matrix(matrix):
                full_matrix = numpy.zeros((original_n, original_n), dtype=matrix.dtype)
                full_matrix[row_start_trim_point:row_end_trim_point, col_start_trim_point:col_end_trim_point] = matrix
                return full_matrix

            # print(region_scores.shape, reconstruct_matrix
            return regions, reconstruct_matrix(region_scores), empirical_distros, reconstruct_matrix(
                region_pvals), reconstruct_matrix(pval_scores), reconstruct_matrix(mask_2d), reconstruct_matrix(
                original_edge_weights)
        else:
            return regions, region_scores, empirical_distros, region_pvals, pval_scores, mask_2d, original_edge_weights

    else:
        return regions


def find_coupled_peaks_1chrom(chrom, min_score=0, max_pval=None, min_size=2, max_size=None,
                              maximization_target='z', trim_input=True, trim_edges=False, gobig=True, tail=None,
                              pvalue_target=DEFAULT_PVALUE_TARGET, start_diagonal=1,
                              quantile_normalize=False, more_smoothing=False,
                              edge_weight_constant=0, edge_weight_power=1,
                              return_debug_data=False, parameter_filter_strength=0, random_seed=None):
    """
    Given a peak correlation matrix, return a list of tuples containing information
    about an optimal set regions such that the number of elements covered by regions with a mean
    value of at least :param:`score_threshold` and a size of at least :param:`min_size` is maximized.

    If :param:`trim-edges` is True, remove points from these regions whose corresponding
    rows fall below :param:`score_threshold`.
    """
    validate_param('maximization_target', maximization_target,
                   ['z', 'p_prod', 'p_sum', 'coverage', 'information', 'wz', 'score'])
    original_n = input_data.shape[0]

    input_data = chrom_data[chrom]  # Get from global inherited data instead of passing as a parameter

    if len(input_data.shape) == 2 and trim_input:
        log_print('{}: trimming {} x {} matrix to remove contiguous NaNs ...'.format(chrom, *input_data.shape), 2)
        row_start_trim_point, row_end_trim_point, col_start_trim_point, col_end_trim_point = compute_matrix_trim_points(
            input_data)
        input_data = input_data[row_start_trim_point:row_end_trim_point, col_start_trim_point:col_end_trim_point]
        log_print('{}: trimmed matrix is now {} x {}'.format(chrom, *input_data.shape), 3)

        input_data = replace_nans_diagonal_means(input_data, start_diagonal=start_diagonal,
                                                 end_diagonal=0)  # ToDo: Handle unsquare trimming results


    else:  # ToDo: perform equivalent trimming for 1D vectors
        row_start_trim_point = 0
    if trim_edges:
        assert min_score > 0  # since edge trimming works on scores of rows, we need a score threshold not just a p-value threshold.
    if len(input_data.shape) > 1:
        assert input_data.shape[0] == input_data.shape[1]
    n = input_data.shape[0]
    if not max_size:
        max_size = n // 2
    else:
        max_size = min(max_size, n)
    assert start_diagonal < min_size
    max_distro_size = max_size + 1  # ToDo: clean up

    if quantile_normalize:
        log_print('{}: quantile-normalizing matrix to standard Gaussian ...'.format(chrom), 2)
        input_data = gaussian_norm(input_data.flatten()).reshape((n, n))

    if len(input_data.shape) == 1:
        log_print('{}: computing means of all subarrays of {}-element vector ...'.format(chrom, n), 2)
        region_scores = simstraps.compute_mean_table_1d(input_data)
        # flat_cell_values = input_data
        if not tail: tail = 'both'
        log_print('{}: constructing null models for regions up to size {} ...'.format(chrom, max_size), 2)
        empirical_distros = simstraps.generate_empirical_distributions_region_means(data=input_data,
                                                                                    max_region_size=max_distro_size,
                                                                                    num_bins=EMPIRICAL_BINS,
                                                                                    max_empirical_size=max_distro_size,
                                                                                    support=(input_data.min(),
                                                                                             input_data.max())
                                                                                    )


    else:
        log_print('{}: computing means of all diagonal square subsets of {} x {} matrix ...'.format(chrom, n, n), 2)
        region_scores = simstraps.compute_mean_table_2d(input_data, start_diagonal=start_diagonal)
        if not tail: tail = 'right'

        # Automatic determination of number of shuffles needed to achieve p-value target based on region sizes. # ToDo: make this better / more principled. Possibly iterative.
        num_shuffles = int(numpy.ceil((1 / pvalue_target / (n - max_size + 1))))

        log_print(
            '{}: constructing null models for regions up to size {} using {} permutations ...'.format(chrom, max_size,
                                                                                                      num_shuffles), 2)

        empirical_distros = simstraps.generate_empirical_distributions_dependent_region_means(matrix=input_data,
                                                                                              num_shuffles=num_shuffles,
                                                                                              min_region_size=min_size,
                                                                                              max_region_size=max_distro_size,
                                                                                              start_diagonal=start_diagonal,
                                                                                              random_seed=random_seed,
                                                                                              filter_window_size=parameter_filter_strength)

    if max_pval is not None:
        log_print('{}: computing -log pvalues ...'.format(chrom), 2)

        if len(input_data.shape) == 1:  # ToDo: Do logsf with flexible tails to avoid this hackiness
            region_pvals = simstraps.compute_pvalues_matrix(data_matrix=region_scores,
                                                            distro_dict=empirical_distros,
                                                            diagonal_start=min_size - 1,
                                                            diagonal_end=max_distro_size,
                                                            tail=tail)
            pval_scores = -numpy.log(numpy.maximum(region_pvals, MIN_PVALUE))
        #             if not return_debug_data:
        #                 del(region_pvals)
        else:
            pval_scores = simstraps.compute_pscores_matrix(data_matrix=region_scores,
                                                           distro_dict=empirical_distros,
                                                           diagonal_start=min_size - 1,
                                                           diagonal_end=max_distro_size,
                                                           tail=tail)

            # log_print('{}: bounding -log pvalues at {} ...'.format(chrom, MIN_PSCORE), 2)
            # print(pval_scores.min(), pval_scores.max())
            # pval_scores = numpy.minimum(pval_scores, MIN_PSCORE)
            region_pvals = numpy.exp(-numpy.minimum(pval_scores, MAX_PSCORE))

    # Apply filters to generate masks

    log_print('{}: applying filters ...'.format(chrom), 2)

    mask_2d = numpy.zeros((n, n), dtype=bool)

    log_print('{}: minimum size: {} ...'.format(chrom, min_size), 3)
    mask_2d[numpy.triu_indices(n, min_size - 1)] = True

    if max_size < n:
        log_print('{}: maximum size: {} ...'.format(chrom, max_size), 3)
        mask_2d[numpy.triu_indices(n, max_size)] = False

    if min_score > 0:
        log_print('{}: minimum absolute score: {} ...'.format(chrom, min_score), 3)
        mask_2d = numpy.logical_and(mask_2d, numpy.greater(numpy.abs(region_scores), min_score))

    if max_pval is not None:
        log_print('{}: maximum p-value: {} ...'.format(chrom, max_pval), 3)
        p_score_threshold = -numpy.log(max_pval)
        mask_2d = numpy.logical_and(mask_2d, numpy.greater(pval_scores, p_score_threshold))

    row_masks, col_masks = create_data_masks(mask_2d)

    # Optimize region sets for given maximization target

    if maximization_target == 'p_prod':
        log_print('{}: minimizing product of p-values'.format(chrom), 2)
        edge_weights = pval_scores.copy()

    elif maximization_target == 'p_sum':
        log_print('{}: minimizing sum of p-values'.format(chrom), 2)
        edge_weights = -region_pvals.copy()

    elif maximization_target == 'coverage':
        log_print('{}: maximizing coverage'.format(chrom), 2)
        edge_weights = create_diagonal_distance_matrix(n).astype(float)

    elif maximization_target == 'score':  # with mean scores this will just pick up minimum size regions so is pretty useless at the moment.
        log_print('{}: maximizing combined mean'.format(chrom), 2)
        edge_weights = region_scores.copy()

    elif maximization_target == 'information':
        log_print('{}: maximizing information content'.format(chrom), 2)
        edge_weights = simstraps.compute_information_matrix(region_scores, empirical_distros,
                                                            diagonal_start=min_size - 1, diagonal_end=max_size)
        print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())

    elif maximization_target == 'z':
        log_print('{}: maximizing standard z score of p-values'.format(chrom), 2)
        edge_weights = region_pvals.copy()
        # print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())
        edge_weights[numpy.equal(region_pvals, 1)] = MAX_PVALUE
        # print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())
        # print(edge_weights)

        edge_weights[numpy.triu_indices(n, 1)] = -scipy.stats.norm().ppf(edge_weights[numpy.triu_indices(n, 1)])
        # print(numpy.isnan(edge_weights).sum(), numpy.isinf(edge_weights).sum(), numpy.isneginf(edge_weights).sum())

    elif maximization_target == 'wz':
        log_print('{}: maximizing coverage-weighted z score of p-values'.format(chrom), 2)
        edge_weights = region_pvals.copy()
        edge_weights[numpy.equal(region_pvals, 1)] = MAX_PVALUE
        edge_weights[numpy.triu_indices(n, 1)] = -scipy.stats.norm().ppf(edge_weights[numpy.triu_indices(n, 1)])
        edge_weights *= create_diagonal_distance_matrix(n).astype(float)

    if more_smoothing:
        log_print('{}: performing additional smoothing of edge weights'.format(chrom), 2)
        edge_weights = simstraps.compute_mean_table_2d(edge_weights, start_diagonal=min_size - 1, end_diagonal=max_size)

    if edge_weight_constant != 0:
        # print(edge_weights.min(), edge_weights.mean(), edge_weights.max())
        log_print('{}: adding {} to all edge weights'.format(chrom, edge_weight_constant), 2)

        edge_weights += edge_weight_constant
        # print(edge_weights.min(), edge_weights.mean(), edge_weights.max())

    if edge_weight_power != 1:
        log_print('{}: raising edge weights to power of {}'.format(chrom, edge_weight_power), 2)
        edge_weights **= edge_weight_power

    #     if not return_debug_data:
    #         del(input_data)
    #         del(mask_2d)
    #         del(pval_scores)

    log_print('{}: computing optimum regions ...'.format(chrom), 2)
    if return_debug_data:
        original_edge_weights = edge_weights.copy()
    score_vec, backtrack = find_maximal_path(edge_weights, row_masks, col_masks, gobig=gobig)
    debug_print(score_vec, backtrack)

    region_coords = decode_backtrack(backtrack)

    if trim_edges:
        log_print('{}: refining region edges ...'.format(chrom), 2)
        region_coords = trim_regions(data=correlation_matrix,
                                     region_coords=region_coords,
                                     min_size=min_size,
                                     edge_trim_threshold=min_score)

    regions = [(start + row_start_trim_point, end + row_start_trim_point, end - start + 1, region_scores[start, end],
                region_pvals[start, end]) for start, end in region_coords[::-1]]

    if return_debug_data:
        if len(input_data.shape) == 2 and trim_input:
            # Reconstruct matrices that correspond to the untrimmed input matrix
            def reconstruct_matrix(matrix):
                full_matrix = numpy.zeros((original_n, original_n), dtype=matrix.dtype)
                full_matrix[row_start_trim_point:row_end_trim_point, col_start_trim_point:col_end_trim_point] = matrix
                return full_matrix

            # print(region_scores.shape, reconstruct_matrix
            return regions, reconstruct_matrix(region_scores), empirical_distros, reconstruct_matrix(
                region_pvals), reconstruct_matrix(pval_scores), reconstruct_matrix(mask_2d), reconstruct_matrix(
                original_edge_weights)
        else:
            return regions, region_scores, empirical_distros, region_pvals, pval_scores, mask_2d, original_edge_weights

    else:
        return regions


def stitch_regions(region_coords, region_scores, region_pvals):
    """
    Given a set of chromosomal regions as a list of start, end, size, score, pvalue tuples,
    return a new list of tuples where adjacent regions with the same sign score are
    joined together, with new scores and p-values obtained from the merged region.
    """
    stitched_regions = []
    in_region = False
    for region_idx in range(len(region_coords) - 1):
        join = region_coords[region_idx + 1][0] - region_coords[region_idx][1] == 1 and region_coords[region_idx][3] * \
               region_coords[region_idx + 1][3] > 0

        #         print(region_idx, region_coords[region_idx], region_coords[region_idx+1], join, in_region)
        if join and not in_region:  # we've just started a stitched region
            combined_region_start = region_coords[region_idx][0]
        elif not join:
            if in_region:
                # we've just finished a stitched region
                combined_region_end = region_coords[region_idx][1]
                combined_region_score = region_scores[combined_region_start, combined_region_end]
                combined_region_pval = region_pvals[combined_region_start, combined_region_end]

                stitched_regions.append((combined_region_start, combined_region_end,
                                         combined_region_end - combined_region_start + 1, combined_region_score,
                                         combined_region_pval))
            else:
                stitched_regions.append(region_coords[region_idx])

        in_region = join
    return stitched_regions

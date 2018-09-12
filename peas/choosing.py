import numpy

from peas.utilities import log_print


def find_maximal_path(edge_weights, row_masks, col_masks, gobig=True):
    """
    Finds a maximum-weight path through the weighted adjacency matrix.

    Returns a list of closed intervals (start and end points are included in the interval)
    on the peak ordinals that represent regions
    that yield an optimal-coverage solution.

    """
    # ToDo: Needs more explicit description of method

    assert edge_weights.shape[0] == edge_weights.shape[1]
    n = edge_weights.shape[0]

    backtrack = numpy.empty(n, dtype=int)
    best_score = numpy.empty(n)

    best_score[0] = float('-Inf')
    backtrack[0] = -1

    for i in range(1, n):
        # print('col_mask: {}'.format(col_masks[i]))
        if len(col_masks[i]) > 0:
            above_vec = edge_weights[col_masks[i], i]

            # Change how we break ties. If gobig is True, numpy.argmax will take the tied
            # score with the smallest index, which will be the most upstream, and therefore
            # yield the biggest region. If False, flip the array so we take the biggest index
            # (and smallest region).
            if gobig:
                best_above_idx = numpy.argmax(above_vec)
            else:
                best_above_idx = len(above_vec) - 1 - numpy.argmax(above_vec[::-1])

            best_above_score = above_vec[best_above_idx]
            best_above_idx = col_masks[i][best_above_idx]
        else:
            best_above_score = float('-Inf')

        previous_score = best_score[i - 1]

        if best_above_score > previous_score:
            best_score[i] = best_above_score
            backtrack[i] = best_above_idx
        else:
            best_score[i] = previous_score
            backtrack[i] = -1

        if best_score[i] > float('-Inf') and i < n - 1 and len(row_masks[i + 1]) > 0:
            edge_weights[i + 1, row_masks[i + 1]] += best_score[i]

    return best_score, backtrack


def decode_backtrack(backtrack):
    """
    Interprets the :param:`backtrack` vector and returns a list of start,end tuples for
    the optimal region coordinates.
    """
    n = len(backtrack)

    regions = []
    total_covered_peaks = 0

    ptr = n - 1

    while not ptr < 0:
        if backtrack[ptr] == -1:
            ptr -= 1
        else:
            start = int(backtrack[ptr])
            end = ptr
            regions.append((start, end))
            ptr = start - 1
            total_covered_peaks += end - start + 1

    log_print('found {} regions covering {} peaks'.format(len(regions), total_covered_peaks), 2)

    return regions


def pick_regions(edge_weights, row_masks, col_masks, gobig=True):
    log_print('computing optimum regions ...', 2)

    score_vec, backtrack = find_maximal_path(edge_weights, row_masks, col_masks, gobig=gobig)

    region_coords = decode_backtrack(backtrack)

    return region_coords


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

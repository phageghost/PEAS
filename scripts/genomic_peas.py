#!python

import argparse
import sys

import peas
from peas import constants


def parse_and_validate_columns(col_param_string, mode):
    column_elements = [col.strip() for col in col_param_string.split(',')]
    try:
        column_elements = [int(col) for col in column_elements]
    except ValueError:
        print('All elements of COLUMNS must be integers.')
        return False

    if not sum([col >= 0 for col in column_elements]) > 0:
        print('All elements of COLUMNS must be non-negative.')
        return False

    if mode == 'matrix':
        if len(column_elements) < 3:
            print('In matrix mode you must specify at least 3 columns to correlate.')
            return False
    else:
        if len(column_elements) not in (1, 2):
            print('In vector mode you must specify either 1 or 2 columns.')
            return False

    return column_elements


# ToDo: Replace hard-coded defaults with imported constants     
def main():
    parser = argparse.ArgumentParser(prog='genomic_peas')
    parser.add_argument('mode', help='operate in vector or matrix mode', type=str, choices=('vector', 'matrix'))
    parser.add_argument('input_file', help='input file', type=str)
    parser.add_argument('input_file_type', help='whether the file is in BED format or HOMER format',
                        choices=('bed', 'homer'))
    parser.add_argument('columns',
                        help='which columns (numbered left-right starting at 0) in the BED file to use as score values. In vector mode, either a single column or a pair of columns can be specified. If a pair is specified, the second column in the pair will be subracted from the first column to generate the vector for PEAS analysis. In matrix mode, 3 or more columns must be specified for use in generating correlation matrices.',
                        type=str)
    parser.add_argument('--tail', '-t',
                        help='look for regions with greater than expected values (right), lower than expected values (left) or either (both)',
                        choices=('right', 'left', 'both'), default='both')
    parser.add_argument('--min-score', '-s', help='minimum region score', type=float,
                        default=constants.DEFAULT_MIN_SCORE)
    parser.add_argument('--pvalue', '-p', help='p-value threshold', type=float,
                        default=constants.DEFAULT_PVALUE_THRESHOLD)
    parser.add_argument('--fdr', '-f', help='FDR threshold', type=float, default=constants.DEFAULT_FDR_THRESHOLD)
    parser.add_argument('--min-size', '-m', help='minimum region size', type=int, default=constants.DEFAULT_MIN_SIZE)
    parser.add_argument('--max-size', '-n', help='maximum region size', type=int, default=constants.DEFAULT_MAX_SIZE)
    parser.add_argument('--alpha', '-a', help='power to apply to edge weights prior to computing optimal regions',
                        type=float, default=2)
    parser.add_argument('--output', '-o', help='output file. If not specified prints output to STDOUT', type=str,
                        default='')

    transform_options = parser.add_argument_group(title='normalization / transformation options',
                                                  description='normalization / transformations to apply to raw data')
    transform_options.add_argument('--col-norm', '-c', action='store_true',
                                   help='normalize each feature column to the same total')
    transform_options.add_argument('--log-transform', '-l', action='store_true',
                                   help='take the log2 of each feature column')
    transform_options.add_argument('--z-transform', '-z', action='store_true',
                                   help='z-score transform each feature column')
    transform_options.add_argument('--pseudocount', '-u', type=float,
                                   default=constants.DEFAULT_PSEUDOCOUNT,
                                   help='pseudocount to add to each feature before log-transforming')                                   

    matrix_options = parser.add_argument_group(title='matrix options',
                                               description='options that only apply when operating in matrix mode')
    matrix_options.add_argument('--ignore-sizes',
                                help='region sizes less than or equal to this number will be ignored for computing scores. This is useful in cases like Hi-C data where adjacent bins may be super-correlated.',
                                type=int, default=1)
    matrix_options.add_argument('--parameter-smoothing-size',
                                help='size of window to consider when smoothing piecewise distribution parameters. If non-zero, will be rounded to nearest odd number.',
                                type=int, default=0)
#    matrix_options.add_argument('--distribution-type',
#                                help='what type of piecewise distribution to fit to the permuted data',
#                                choices=constants.NULL_DISTRIBUTIONS_BY_NAME.keys(),
#                               default=constants.DEFAULT_NULL_DISTRIBUTION)
    matrix_options.add_argument('--random-seed', '-r',
                                help='value to use to initialize psuedorandom number generator. If not specified a new seed will be chosen each time',
                                type=float, default=None)

    vector_options = parser.add_argument_group(title='vector options',
                                               description='options that only apply when operating in vector mode')
    vector_options.add_argument('--bins', '-b',
                                help='how many bins to use for approximating the empirical null distributions',
                                type=str,
                                default='auto')

    args = parser.parse_args()

    feature_columns = parse_and_validate_columns(args.columns, args.mode)
    if feature_columns:
        if args.mode == 'vector':
            peas.genomic_regions.find_genomic_region_crds_vector(peak_filename=args.input_file,
                                                                 peak_file_format=args.input_file_type,
                                                                 feature_columns=feature_columns,
                                                                 rip_norm=args.col_norm,
                                                                 znorm=args.z_transform,
                                                                 log_transform=args.log_transform,
                                                                 pseudocount=args.pseudocount,
                                                                 tail=args.tail, min_score=args.min_score,
                                                                 pvalue=args.pvalue, fdr=args.fdr,
                                                                 min_size=args.min_size,
                                                                 max_size=args.max_size, alpha=args.alpha,
                                                                 bins=args.bins,
                                                                 output_filename=args.output)
        else:
            peas.genomic_regions.find_genomic_region_crds_matrix(peak_filename=args.input_file,
                                                                 peak_file_format=args.input_file_type,
                                                                 feature_columns=feature_columns,
                                                                 rip_norm=args.col_norm,
                                                                 znorm=args.z_transform,
                                                                 log_transform=args.log_transform,
                                                                 pseudocount=args.pseudocount,
                                                                 tail=args.tail, min_score=args.min_score,
                                                                 pvalue=args.pvalue, fdr=args.fdr,
                                                                 min_size=args.min_size,
                                                                 max_size=args.max_size, alpha=args.alpha,
                                                                 output_filename=args.output,
                                                                 start_diagonal=args.ignore_sizes,
                                                                 parameter_smoothing_window_size=args.parameter_smoothing_size,
                                                                 null_distribution_type=constants.DEFAULT_NULL_DISTRIBUTION,
                                                                 random_seed=args.random_seed)


if __name__ == '__main__':
    sys.exit(main())

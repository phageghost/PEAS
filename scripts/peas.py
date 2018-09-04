#!python

import sys
import argparse
import peas

def validate_columns(col_param_string, mode):
    column_elements = [col.strip() for col in col_param_string.split(',')]
    try:
        column_elements = [int(col) for col in column_elements]
    except ValueError:
        print('All elements of COLUMNS must be integers.')
        return False
    
    if not sum([col >= 0 for col in column_elements]) > 0:
        print ('All elements of COLUMNS must be non-negative.')
        return False
        
    if mode == 'matrix':
        if len(column_elements) < 3:
            print('In matrix mode you must specify at least 3 columns to correlate.')
            return False
    else:
        if len(column_elements) not in (1,2):
            print('In vector mode you must specify either 1 or 2 columns.')
            return False
    
    return True

    
def main():
    parser = argparse.ArgumentParser(prog='peas')
    parser.add_argument('mode', help='operate in vector or matrix mode', type=str, choices=('vector', 'matrix'))
    parser.add_argument('input_file', help='input file', type=argparse.FileType('r'))
    parser.add_argument('columns', help='which columns (numbered left-right starting at 0) in the BED file to use as score values. In vector mode, either a single column or a pair of columns can be specified. If a pair is specified, the second column in the pair will be subracted from the first column to generate the vector for PEAS analysis. In matrix mode, 3 or more columns must be specified for use in generating correlation matrices.')
    
    # Add options for score transformations
    
    parser.add_argument('--tail', '-t', help='look for regions with greater than expected values (right), lower than expected values (left) or either (both)', choices=('right', 'left', 'both'), default='both')
    parser.add_argument('--min-score', '-s', help='minimum region score', type=float, default=0.0)
    parser.add_argument('--pvalue', '-p', help='p-value threshold', type=float, default=1e-3)
    parser.add_argument('--fdr', '-f', help='FDR threshold', type=float, default=0.05)
    parser.add_argument('--min-size', '-m', help='minimum region size', type=int, default=2)
    parser.add_argument('--max-size', '-n', help='maximum region size', type=int, default=0)
    parser.add_argument('--alpha', '-a', help='power to apply to edge weights prior to computing optimal regions', type=float, default=2)
    
    matrix_options = parser.add_argument_group(title='matrix options', description='options that only apply when operating in matrix mode')
    matrix_options.add_argument('--parameter-smoothing-size', help='size of window to consider when smoothing piecewise distribution parameters', type=str, default=5)
    matrix_options.add_argument('--distribution-type', help='what type of piecewise distribution to fit to the permuted data', choices=('pw_power', 'pw_linear'))
    
    vector_options = parser.add_argument_group(title='vector options', description='options that only apply when operating in vector mode')
    vector_options.add_argument('--bins', '-b', help='how many bins to use for approximating the empirical null distributions', type=str, default='auto')
    
    args = parser.parse_args()
    
    if validate_columns(args.columns, args.mode):
    
        print(args)


    # peas.main.find_ropes()


if __name__ == '__main__':
    sys.exit(main())


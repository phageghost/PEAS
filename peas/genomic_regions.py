import numpy
import pandas
from statsmodels.stats.multitest import multipletests

from peas.utilities import log_print, pseudolinearize
from . import constants
from . import interface


def print_df_full(df, sep='\t'):
    """
    Display the contents of a pandas.DataFrame to screen, separated
    by :param sep:
    """
    print(sep.join(df.columns))
    for row_tuple in df.itertuples():
        print(sep.join([str(val) for val in row_tuple]))


def load_and_parse_bed_file(bed_fname, feature_columns):
    """
    Assumes columns are: chrom, chromStart, chromEnd, name, score, strand
    """
    assert sum([col > 3 for col in feature_columns]) == len(feature_columns)

    bed_df = pandas.read_csv(bed_fname, sep='\t', header=None)

    bed_df = bed_df.sort_values(by=[0, 1])

    annotations = bed_df.iloc[:, :4]
    features = bed_df.iloc[:, feature_columns]

    return annotations, features


def load_and_parse_peak_file(peak_fname, feature_columns):
    """
    Assumes columns are: peak_id, Chr, Start, End, Strand, Peak Score, Focus Ratio/Region Size, data columns
    """
    assert sum([col > 4 for col in feature_columns]) == len(feature_columns)
    peak_df = pandas.read_csv(peak_fname, index_col=0, sep='\t')
    chrom_column_heading = peak_df.columns[0]
    start_column_heading = peak_df.columns[1]

    peak_df = peak_df.sort_values(by=[chrom_column_heading, start_column_heading])

    annotations = peak_df.iloc[:, :4]
    features = peak_df.iloc[:, numpy.array(feature_columns) - 1]  # account for index column

    return annotations, features


def normalize_features(features, rip_norm=constants.DEFAULT_RIP_NORM, znorm=constants.DEFAULT_ZNORM,
                        log_transform=constants.DEFAULT_LOG_TRANSFORM, pseudocount=constants.DEFAULT_PSEUDOCOUNT):
    """
    Given a pandas DataFrame with peaks for each condition in columns,
    normalize the column vectors according to the given flags.
    """
    # ToDo: add option for pseudocount transform
    if rip_norm:
        log_print('Normalizing by reads in regions ...', 3)
        condition_means = features.mean(axis=0)
        features /= condition_means / condition_means.mean()
        
    if pseudocount > 0:
        log_print('Adding pseudocount of {}'.format(pseudocount), 3)
        features += pseudocount     
        
    if log_transform:
        log_print('Log transforming ...', 3)
        features = numpy.log2(features + 1)
        if pseudocount > 0:
            features -= numpy.log2(pseudocount)

    if znorm:
        log_print('Z-score transforming ...', 3)
        condition_means = features.mean(axis=0)
        condition_stds = features.std(axis=0)
        features = (features - condition_means) / condition_stds
        
    return features


def find_genomic_region_crds_vector(peak_filename, peak_file_format, feature_columns, output_filename='',
                                    rip_norm=constants.DEFAULT_RIP_NORM, znorm=constants.DEFAULT_ZNORM,
                                    log_transform=constants.DEFAULT_LOG_TRANSFORM, 
                                    pseudocount=constants.DEFAULT_PSEUDOCOUNT,                                    
                                    tail=constants.DEFAULT_TAIL,
                                    min_score=constants.DEFAULT_MIN_SCORE, pvalue=constants.DEFAULT_PVALUE_THRESHOLD,
                                    fdr=constants.DEFAULT_FDR_THRESHOLD, min_size=constants.DEFAULT_MIN_SIZE,
                                    max_size=constants.DEFAULT_MAX_SIZE, alpha=constants.DEFAULT_ALPHA,
                                    bins=constants.DEFAULT_BINS):
    if peak_file_format == 'bed':
        annotations, features = load_and_parse_bed_file(bed_fname=peak_filename, feature_columns=feature_columns)
    else:
        annotations, features = load_and_parse_peak_file(peak_fname=peak_filename, feature_columns=feature_columns)

    log_print('Loaded {} genomic regions from {}'.format(annotations.shape[0], peak_filename))

    assert min_size >= 2
    assert 0 < pvalue <= 1
    assert 0 <= min_score

    features = normalize_features(features=features, rip_norm=rip_norm, znorm=znorm,
                                  log_transform=log_transform, pseudocount=pseudocount)

    if len(feature_columns) == 1:
        features = features.iloc[:, 0]
    else:
        assert len(feature_columns) == 2
        assert feature_columns[0] != feature_columns[1]
        if feature_columns[1] < feature_columns[0]:
            features = features.iloc[:, 1] - features.iloc[:, 0]
        else:
            features = features.iloc[:, 0] - features.iloc[:, 1]

    total_regions = 0
    region_dfs = []
    for chrom, chrom_annotations in annotations.groupby(annotations.columns[0]):
        log_print('processing {} elements in chromosome {}'.format(chrom_annotations.shape[0], chrom), 1)
        chrom_vector = features.loc[chrom_annotations.index].values

        this_chrom_peas = interface.find_peas_vector(input_vector=chrom_vector, min_score=min_score,
                                                     max_pval=pvalue, min_size=min_size, max_size=max_size,
                                                     maximization_target=constants.DEFAULT_MAXIMIZATION_TARGET,
                                                     tail=tail,
                                                     bins=bins,
                                                     quantile_normalize=False,
                                                     edge_weight_power=alpha,
                                                     gobig=True)

        this_chrom_peas_df = generate_bed_df(this_chrom_peas, chrom_annotations)
        total_regions += this_chrom_peas_df.shape[0]
        region_dfs.append(this_chrom_peas_df)

    if total_regions > 0:
        all_regions_df = pandas.concat([region_df for region_df in region_dfs if region_df.shape[0] > 0], axis=0)

        if fdr is not None:
            passfail, qvals, _, _ = multipletests(all_regions_df.pval, alpha=fdr, method='fdr_bh')
            all_regions_df['qval'] = qvals
            all_regions_df = all_regions_df.loc[passfail]
            log_print('{} PEAs passed FDR threshold of {}.'.format(all_regions_df.shape[0], fdr), 1)

        if all_regions_df.shape[0] == 0:
            print('no PEAs passed the FDR threshold!')
        else:
            if output_filename:
                # write_ucsc_bed_file(filename=output_filename, bed_df=all_regions_df, track_name='PEAs', description='')
                all_regions_df.to_csv(output_filename, sep='\t')
            else:
                print_df_full(all_regions_df)


def find_genomic_region_crds_matrix(peak_filename, peak_file_format, feature_columns, output_filename='',
                                    rip_norm=constants.DEFAULT_RIP_NORM, znorm=constants.DEFAULT_ZNORM,
                                    log_transform=constants.DEFAULT_LOG_TRANSFORM, 
                                    pseudocount=constants.DEFAULT_PSEUDOCOUNT,
                                    tail=constants.DEFAULT_TAIL,
                                    min_score=constants.DEFAULT_MIN_SCORE, pvalue=constants.DEFAULT_PVALUE_THRESHOLD,
                                    fdr=constants.DEFAULT_FDR_THRESHOLD, min_size=constants.DEFAULT_MIN_SIZE,
                                    max_size=constants.DEFAULT_MAX_SIZE, alpha=constants.DEFAULT_ALPHA,
                                    start_diagonal=constants.DEFAULT_START_DIAGONAL,
                                    parameter_smoothing_window_size=constants.DEFAULT_PARAMETER_SMOOTHING_WINDOW_SIZE,
                                    null_distribution_type=constants.DEFAULT_NULL_DISTRIBUTION,
                                    random_seed=None,
                                    ):
    if peak_file_format == 'bed':
        annotations, features = load_and_parse_bed_file(bed_fname=peak_filename, feature_columns=feature_columns)
    else:
        annotations, features = load_and_parse_peak_file(peak_fname=peak_filename, feature_columns=feature_columns)

    log_print('loaded {} genomic regions from {}'.format(annotations.shape[0], peak_filename))

    assert min_size >= 2
    assert 0 < pvalue <= 1
    assert 0 <= min_score

    features = normalize_features(features=features, rip_norm=rip_norm, znorm=znorm,
                                  log_transform=log_transform, pseudocount=pseudocount)

    total_regions = 0
    region_dfs = []
    for chrom, chrom_annotations in annotations.groupby(annotations.columns[0]):
        log_print('processing {} elements in chromosome {}'.format(chrom_annotations.shape[0], chrom), 1)
        chrom_matrix = features.loc[chrom_annotations.index]
        log_print('correlating ...', 2)
        chrom_corrs = chrom_matrix.T.corr().values

        this_chrom_peas = interface.find_peas_matrix(input_matrix=chrom_corrs, min_score=min_score,
                                                     max_pval=pvalue, min_size=min_size, max_size=max_size,
                                                     maximization_target=constants.DEFAULT_MAXIMIZATION_TARGET,
                                                     tail=tail,
                                                     quantile_normalize=False,
                                                     edge_weight_power=alpha,
                                                     pvalue_target=constants.DEFAULT_PVALUE_TARGET,
                                                     max_pvalue_cv=constants.DEFAULT_PVALUE_CV,
                                                     num_shuffles='auto',
                                                     null_distribution_class=null_distribution_type,
                                                     start_diagonal=start_diagonal,
                                                     parameter_smoothing_method=constants.DEFAULT_PARAMETER_SMOOTHING_METHOD,
                                                     parameter_filter_strength=parameter_smoothing_window_size,
                                                     random_seed=random_seed,
                                                     gobig=True
                                                     )

        this_chrom_peas_df = generate_bed_df(this_chrom_peas, chrom_annotations)
        total_regions += this_chrom_peas_df.shape[0]
        region_dfs.append(this_chrom_peas_df)

    if total_regions > 0:
        all_regions_df = pandas.concat([region_df for region_df in region_dfs if region_df.shape[0] > 0], axis=0)

        if fdr is not None:
            passfail, qvals, _, _ = multipletests(all_regions_df.pval, alpha=fdr, method='fdr_bh')
            all_regions_df['qval'] = qvals
            all_regions_df = all_regions_df.loc[passfail]
            log_print('{} PEAs passed FDR threshold of {}.'.format(all_regions_df.shape[0], fdr), 1)

        if all_regions_df.shape[0] == 0:
            print('no PEAs passed the FDR threshold!')
        else:
            if output_filename:
                # write_ucsc_bed_file(filename=output_filename, bed_df=all_regions_df, track_name='PEAs', description='')
                all_regions_df.to_csv(output_filename, sep='\t')
            else:
                print_df_full(all_regions_df)


def generate_bed_df(regions, annotations):
    """
    Takes a list of region data and returns a DataFrame of peak information suitable for writing to a BED file.
    """
    all_peak_names = list(annotations.index)
    region_list = []

    for start, end, size, score, pval in regions:
        peak_names = all_peak_names[start:end + 1]
        compound_peak_name = '_'.join([str(peak_name) for peak_name in peak_names])

        overall_start = annotations.iloc[start, 1]
        overall_end = annotations.iloc[end, 2]

        # sanity check
        assert annotations.iloc[start, 0] == annotations.iloc[
            end, 0], 'First region peak chromosome {} doesn\'t match end peak chromosome {}'.format \
            (annotations.iloc[start, 0], annotations.iloc[end, 0])
        assert overall_start < overall_end, 'Invalid region start and end positions: {}, {}'.format(overall_start,
                                                                                                    overall_end)

        chrom = annotations.iloc[start, 0]

        region_list.append((chrom, overall_start, overall_end, compound_peak_name, score, '+', pval))

    if region_list:
        region_df = pandas.DataFrame(region_list)
        region_df.columns = ('chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'pval')
        region_df.index = region_df['name']

    else:
        # Return an empty DataFrame
        region_df = pandas.DataFrame(columns=('chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'pval'))

    return region_df


def write_ucsc_bed_file(filename, bed_df, track_name, description=''):
    """
    Writes the contents of :param:bed_df to :param:`filename` as a BED format
    file with a header line as required by UCSC genome browser.
    """
    ucsc_formatted_bed_df = bed_df.loc[:, ('chrom', 'chromStart', 'chromEnd', 'name', 'score',
                                           'strand')]  # trim off pvalue and qvalue columns to avoid confusing UCSC browser.
    with open(filename, 'wt') as out_file:
        out_file.write('track name={} description="{}"\n'.format(track_name, description))
        ucsc_formatted_bed_df.to_csv(out_file, sep='\t', header=False, index=False)

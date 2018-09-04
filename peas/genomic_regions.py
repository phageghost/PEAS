import numpy
import pandas
from statsmodels.stats.multitest import multipletests

from peas.utilities import log_print
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


def load_and_parse_bed_file(bed_fname, value_columns):
    """
    Assumes columns are: chrom, chromStart, chromEnd, name, score, strand
    """
    assert sum([col > 3 for col in value_columns]) == value_columns

    bed_df = pandas.read_csv(bed_fname, sep='\t')

    bed_df = bed_df.sort_values(by=[0, 1])

    annotations = bed_df.iloc[:, :4]
    features = bed_df.iloc[:, value_columns]

    return annotations, features


def load_and_parse_peak_file(peak_fname, value_columns):
    """
    Assumes columns are: peak_id, Chr, Start, End, Strand, Peak Score, Focus Ratio/Region Size, data columns
    """
    assert sum([col > 4 for col in value_columns]) == value_columns
    peak_df = pandas.read_csv(peak_fname, index_col=0, sep='\t')

    peak_df = peak_df.sort_values(by=[0, 1])

    annotations = peak_df.iloc[:, :4]
    features = peak_df.iloc[:, value_columns]

    return annotations, features


def normalize_features(features, value_columns, rip_norm=True, znorm=False, log_transform=True):
    """
    Given a pandas DataFrame with peaks for each condition in columns,
    normalize the column vectors according to the given flags.
    """
    # ToDo: add option for pseudocount transform
    if rip_norm:
        log_print('Normalizing by reads in regions ...', 3)
        condition_means = features.iloc[:, value_columns].mean(axis=0)
        features.iloc[:, value_columns] /= condition_means / condition_means.mean()

    if log_transform:
        log_print('Log transforming ...', 3)
        features.iloc[:, value_columns] = numpy.log2(features.iloc[:, value_columns] + 1)

    if znorm:
        log_print('Z-score transforming ...', 3)
        condition_means = features.iloc[:, value_columns].mean(axis=0)
        condition_stds = features.iloc[:, value_columns].std(axis=0)
        features.iloc[:, value_columns] = (features.iloc[:,
                                           value_columns] - condition_means) / condition_stds

    return features


# ToDo: Add default values using module-wide constants.
def find_genomic_region_crds_vector(peak_filename, peak_file_format, output_filename, feature_columns, rip_norm, znorm,
                                    log_transform, tail, min_score, pvalue, fdr, min_size, max_size, alpha, bins):
    if peak_file_format == 'bed':
        annotations, features = load_and_parse_bed_file(bed_fname=peak_filename, value_columns=feature_columns)
    else:
        annotations, features = load_and_parse_peak_file(peak_fname=peak_filename, value_columns=feature_columns)

    features = normalize_features(features=features, value_columns=feature_columns, rip_norm=rip_norm, znorm=znorm,
                                  log_transform=log_transform)

    print(features.shape)

    if len(feature_columns) == 1:
        features = features.iloc[:, 0]
    else:
        features = features.iloc[:, feature_columns[0]] - features.iloc[:, feature_columns[1]]

    total_regions = 0
    region_dfs = []
    for chrom, chrom_annotations in annotations.groupby(1):
        log_print('Processing {} elements in chromosome {}'.format(chrom_annotations.shape[0], chrom), 2)
        chrom_vector = features.loc[chrom_annotations.index]
        this_chrom_ropes = interface.find_ropes_vector(input_vector=chrom_vector, min_score=min_score,
                                                       max_pval=pvalue, min_size=min_size, max_size=max_size,
                                                       maximization_target=constants.DEFAULT_MAXIMIZATION_TARGET,
                                                       tail=tail,
                                                       quantile_normalize=False,
                                                       edge_weight_power=alpha,
                                                       gobig=True)

    this_chrom_ropes_df = generate_bed_df(this_chrom_ropes, chrom_annotations)
    total_regions += this_chrom_ropes_df.shape[0]
    region_dfs.append(this_chrom_ropes_df)

    if total_regions > 0:
        all_regions_df = pandas.concat([region_df for region_df in region_dfs if region_df.shape[0] > 0], axis=0)

        if fdr is not None:
            passfail, qvals, _, _ = multipletests(all_regions_df.pval, alpha=fdr, method='fdr_bh')
            all_regions_df['qval'] = qvals
            all_regions_df = all_regions_df.loc[passfail]
            log_print('{} ROPES passed FDR threshold of {}.'.format(all_regions_df.shape[0], fdr), 1)

        if all_regions_df.shape[0] == 0:
            print('No ROPES passed the FDR threshold!')
        else:
            if output_filename:
                write_ucsc_bed_file(filename=output_filename, bed_df=all_regions_df, track_name='ROPES', description='')
            else:
                print_df_full(all_regions_df)


# def correlation_matrix_worker(params):
#     print('*' * 80)
#     print('Global namespace: {}'.format(dir()))
#     print('*' * 80)
#
#     fname_prefix, chrom, this_chrom_annotated = params
#
#     this_chrom_annotated = this_chrom_annotated.sort_values(by=['Chr', 'Start'])
#     this_chrom_data = this_chrom_annotated.iloc[:, 6:]
#     log_print('Pairwise correlating {} regions in chromosome {} ... '.format(this_chrom_data.shape[0],
#                                                                              chrom), 2)
#
#     this_chrom_corrs = this_chrom_data.T.corr()
#
#     if fname_prefix:
#         this_fname = '{}_{}.txt'.format(fname_prefix, chrom)
#         log_print('Saving chromosome {} correlations to {} ... '.format(chrom, this_fname), 2)
#         this_chrom_corrs.to_csv(this_fname, sep='\t')
#         chrom_data[chrom] = this_chrom_corrs
#     return chrom
#
#
# def generate_correlation_matrices_from_annotated_peaks_multicore(annotated_peak_df,
#                                                                  fname_prefix='',
#                                                                  num_cores=16):
#     print('*' * 80)
#     print('Global namespace: {}'.format(dir()))
#     print('*' * 80)
#
#     params = []
#     for chrom, this_chrom_annotated in annotated_peak_df.groupby('Chr'):
#         chrom = chrom.strip()
#         if chrom != 'chrM':
#             params.append((fname_prefix, chrom, this_chrom_annotated))
#     corr_pool = multiprocessing.Pool(num_cores)
#     _ = corr_pool.map(correlation_matrix_worker, params)
#     log_print('Done correlating all chromosomes.', 1)
#     corr_pool.close()
#     corr_pool.join()
#
#
# # def sort_df(df):
# #     sort_index = toolbox.numerical_string_sort(df.index)
# #     return df.loc[sort_index, sort_index]
#
# def find_crds_worker(params):
#     chrom, annotations, min_score, max_pval, min_size, max_size, edge_weight_power, pvalue_target, trim_edges, maximization_target, gobig, random_seed = params
#
#     if len(chrom_data[chrom].shape) > 1:
#         assert chrom_data[chrom].shape[0] == chrom_data[chrom].shape[1]
#     num_features = chrom_data[chrom].shape[0]
#
#     this_chrom_regions = find_coupled_peaks_1chrom(chrom,
#                                                    min_score=min_score,
#                                                    max_pval=max_pval,
#                                                    min_size=min_size,
#                                                    max_size=max_size,
#                                                    maximization_target=maximization_target,
#                                                    pvalue_target=pvalue_target,
#                                                    edge_weight_power=edge_weight_power,
#                                                    trim_edges=trim_edges,
#                                                    gobig=gobig,
#                                                    random_seed=random_seed)
#     regions_found = len(this_chrom_regions)
#
#     log_print('{}: extracting annotations ...'.format(chrom), 2)
#
#     chrom_df = generate_bed_df(this_chrom_regions, annotations)
#
#     log_print('{}: done.'.format(chrom), 2)
#
#     return chrom, regions_found, num_features, chrom_df


def generate_bed_df(regions, annotations):
    """
    Takes a list of region data and returns a Dataframe of peak information suitable for writing to a BED file.
    """
    all_peak_names = list(annotations.index)
    region_list = []

    for start, end, size, score, pval in regions:
        peak_names = all_peak_names[start:end + 1]
        compound_peak_name = '_'.join(peak_names)

        overall_start = annotations.iloc[start].loc['Start']
        overall_end = annotations.iloc[end].loc['End']

        # sanity check
        assert annotations.iloc[start].loc['Chr'] == annotations.iloc[end].loc
        ['Chr'], 'First region peak chromosome {} doesn\'t match end peak chromosome {}'.format \
            (annotations.iloc[start].loc['Chr'], annotations.iloc[end].loc['Chr'])
        assert overall_start < overall_end, 'Invalid region start and end positions: {}, {}'.format(overall_start,
                                                                                                    overall_end)

    chrom = annotations.iloc[start].loc['Chr']

    region_list.append((chrom, overall_start, overall_end, compound_peak_name, score, '+', pval))

    if region_list:
        region_df = pandas.DataFrame(region_list)
        region_df.columns = ('chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'pval')
        region_df.index = region_df['name']

    else:
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

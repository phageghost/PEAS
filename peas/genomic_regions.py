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

debug_print(region_df)

return region_df


def make_UCSC_bed_file(filename, bed_df, track_name, description=''):
    """
    Writes the contents of :param:bed_df to :param:`filename` as a BED format
    file with a header line as required by UCSC genome browser.
    """
    ucsc_formatted_bed_df = bed_df.loc[:, ('chrom', 'chromStart', 'chromEnd', 'name', 'score',
                                           'strand')]  # trim off pvalue and qvalue columns to avoid confusing UCSC browser.
    with open(filename, 'wt') as out_file:
        out_file.write('track name={} description="{}"\n'.format(track_name, description))
        ucsc_formatted_bed_df.to_csv(out_file, sep='\t', header=False, index=False)

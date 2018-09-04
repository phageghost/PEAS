import pandas


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
        
        
        
        
def parse_bed_file(bed_fname, 
        

        
        
        
def find_vector_crds(input_data, 
                        annotations,
                        min_score=None,
                        max_pval=0.05,
                        min_size=2, 
                        max_size=None, 
                        edge_weight_power=1,
                        tail='both',
                        fdr=0.05, 
                        trim_edges=False, gobig=True, 
                        num_cores=24,
                        maximization_target='p_prod',
                        pvalue_target=DEFAULT_PVALUE_TARGET,
                        random_seed=None):

    region_dfs = []
    total_peaks = 0
    total_regions = 0
    
    # Convert data to dict keyed by chrom. Perform correlations if 2D features
    annotations_by_chrom = {}
    for chrom, chrom_anno in annotations.groupby('Chr'):
        annotations_by_chrom[chrom] = chrom_anno.sort_values('Start')
        
#     return annotations_by_chrom
    
    chrom_data = {}
    print('*'*80)
    print('Global namespace: {}'.format(dir()))
    print('*'*80)
    
    if len(input_data.shape) > 1:
        log_print('Performing pairwise correlations...', 1)
        generate_correlation_matrices_from_annotated_peaks_multicore(annotations, fname_prefix='', num_cores=num_cores)
    else:
        for chrom in annotations_by_chrom:
            chrom_data[chrom] = input_data.loc[annotations_by_chrom[chrom].index]
        
    params = []
        
    for chrom in sorted(input_data_by_chrom):
        chrom = chrom.strip()
        if chrom is not 'chrM':
            params.append([chrom,
                           annotations_by_chrom[chrom], 
                           min_score,
                           max_pval,
                           min_size,
                           max_size, 
                           edge_weight_power,
                           pvalue_target,
                           trim_edges,
                           maximization_target, 
                           gobig,
                           random_seed])
        
    total_regions = 0
    total_peaks = 0
    region_dfs = []
    crd_pool = multiprocessing.Pool(num_cores)
    
    for chrom, regions_found, peaks_found, chrom_df in crd_pool.map(find_crds_worker, params):
        total_regions += regions_found
        total_peaks += peaks_found
        region_dfs.append(chrom_df)
    crd_pool.close()
    crd_pool.join()
    
    log_print('Found {} CRDs in {} total pairwise correlations.'.format(total_regions, total_peaks), 1)
    
    if total_regions > 0:
        all_regions_df = pandas.concat([region_df for region_df in region_dfs if region_df.shape[0] > 0], axis=0)
        
        if fdr is not None:
            passfail, qvals, _, _  = multipletests(all_regions_df.pval, alpha=fdr, method='fdr_bh')
            all_regions_df['qval'] = qvals
            all_regions_df = all_regions_df.loc[passfail]
            log_print('{} CRDs passed FDR threshold of {}.'.format(all_regions_df.shape[0], fdr), 1)
        
        return all_regions_df    
        
    else:
        return None    

def find_crds_worker(params):
    chrom, annotations, min_score, max_pval, min_size, max_size, edge_weight_power, pvalue_target, trim_edges, maximization_target, gobig, random_seed = params
    
    if len(chrom_data[chrom].shape) > 1:
        assert chrom_data[chrom].shape[0] == chrom_data[chrom].shape[1]
    num_features = chrom_data[chrom].shape[0]

   
    this_chrom_regions = find_coupled_peaks_1chrom(chrom,
                                            min_score=min_score,
                                            max_pval=max_pval,
                                            min_size=min_size,
                                            max_size=max_size,
                                            maximization_target=maximization_target,
                                            pvalue_target=pvalue_target,
                                            edge_weight_power=edge_weight_power,
                                            trim_edges=trim_edges,
                                            gobig=gobig,
                                            random_seed=random_seed)  
    regions_found = len(this_chrom_regions)

    log_print('{}: extracting annotations ...'.format(chrom), 2)

    chrom_df = generate_bed_df(this_chrom_regions, annotations)

    log_print('{}: done.'.format(chrom), 2)
    
    return chrom, regions_found, num_features, chrom_df
        
import numpy

import empdist
from . import scoring

MAX_PSCORE = 744.44007192138122
MIN_PVALUE = numpy.exp(-MAX_PSCORE)
MAX_PVAL = 1 - 1e-100
DEFAULT_PVALUE_TARGET = 1e-4
DEFAULT_MAX_PVALUE_CV = 0.05
DEFAULT_PSEUDOCOUNT = 0
DEFAULT_MIN_SIZE = 3
DEFAULT_MAX_SIZE = 500
DEFAULT_MAX_SIZE_FACTOR = 4
DEFAULT_MIN_SCORE = 0
DEFAULT_PVALUE_THRESHOLD = 1e-3
DEFAULT_FDR_THRESHOLD = 0.05
DEFAULT_RIP_NORM = True
DEFAULT_ZNORM = False
DEFAULT_LOG_TRANSFORM = True
DEFAULT_TAIL = 'both'
DEFAULT_ALPHA = 2
DEFAULT_BINS = 'auto'
VECTOR_SCORE_FUNCS_BY_NAME = {'mean': empdist.helper_funcs.predict_distributions_independent_means,
                              'sum': empdist.helper_funcs.predict_distributions_independent_sums,
                              'min': empdist.helper_funcs.predict_distributions_independent_mins,
                              'max': empdist.helper_funcs.predict_distributions_independent_maxes}
MATRIX_SCORING_FUNCS_BY_NAME = {'sum': scoring.compute_sum_table_2d,
                                'mean': scoring.compute_mean_table_2d,
                                'min': scoring.compute_min_table_2d,
                                'max': scoring.compute_max_table_2d}
DEFAULT_SCORE_FUNC = 'mean'
NULL_DISTRIBUTIONS_BY_NAME = {'hybrid': empdist.HybridDistribution}
DEFAULT_NULL_DISTRIBUTION = 'hybrid'
DEFAULT_PARAMETER_SMOOTHING_METHOD = 'savgol'
DEFAULT_PARAMETER_SMOOTHING_WINDOW_SIZE = 0
DEFAULT_MAXIMIZATION_TARGET = 'p_prod'
DEFAULT_START_DIAGONAL = 1
USE_C = True

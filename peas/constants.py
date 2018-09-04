import empdist

from peas.fitapproxdistros.constants import *
from . import scoring

DEFAULT_PVALUE_TARGET = 1e-6
MAX_PSCORE = 744.44007192138122
MIN_PVALUE = numpy.exp(-MAX_PSCORE)
MAX_PVAL = 1 - 1e-100
DEFAULT_PVALUE_CV = 0.05
DEFAULT_PSEUDOCOUNT = 0

VECTOR_SCORE_FUNCS_BY_NAME = {'mean': empdist.helper_funcs.predict_distributions_independent_means,
                              'sum': empdist.helper_funcs.predict_distributions_independent_sums,
                              'min': empdist.helper_funcs.predict_distributions_independent_mins,
                              'max': empdist.helper_funcs.predict_distributions_independent_maxes}

MATRIX_SCORING_FUNCS_BY_NAME = {'sum': scoring.compute_sum_table_2d,
                                'mean': scoring.compute_mean_table_2d,
                                'min': scoring.compute_min_table_2d,
                                'max': scoring.compute_max_table_2d}

NULL_DISTRIBUTIONS_BY_NAME = {'pw_power': distributions.PiecewiseApproxPower,
                              'pw_linear': distributions.PiecewiseApproxLinear}

DEFAULT_PARAMETER_SMOOTHING_METHOD = 'savgol'
SAVGOL_DEFAULT_WINDOW_SIZE = 5
DEFAULT_NULL_DISTRIBUTION = 'pw_power'
DEFAULT_MAXIMIZATION_TARGET = 'p_prod'

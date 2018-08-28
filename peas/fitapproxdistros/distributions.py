import numpy
import scipy.stats

from empdist.empirical_pval import compute_empirical_pvalue,  determine_p_cutoff, compute_empirical_quantile
from scipy.optimize import curve_fit

DEFAULT_MAX_EMPIRICAL_SIZE = 50
DEFAULT_SUPPORT_QUANTILE = 0.9


class PiecewiseApproxLinear:
    """
    Stub class for an empirical distribution with methods to:
        1. Fit a piecewise linear function to the log-survival function of a data sample
        2. Compute the value of the log-survival function for given x.
    """

    def __init__(self, inflection_point, slope):
        self.inflection_point = inflection_point
        self.slope = slope

    @staticmethod
    def _piecewise_logsf(x, inflection_point, slope):
        """
        A piecewise linear function that = 0 for all x < :param:`inflection_point`
            and rises linearly with slope :param:`slope` for all points > :param:`inflection_point`
        """
        return numpy.piecewise(x, [x < inflection_point], [lambda x: 0, lambda x: slope * (x - inflection_point)])

    @classmethod
    def fit(cls, data, is_sorted=False, max_pvalue_std_error=0.05, interp_points=50,
            initial_inflection_point=None, initial_slope=500):
        if not is_sorted:
            data = numpy.sort(data)

        min_val, max_val = data.min(), data.max()
        data_mean = data.mean()
        endpoint = compute_empirical_quantile(data, 1 - determine_p_cutoff(n=len(data)), is_sorted=True)

        if endpoint <= data_mean:
            raise ValueError(
                'Minimum data value that meets target p-value error threshold of {} is {}, which is below the mean of {}, therefore piecewise linear approximation will not work. Try generating more empirical samples or increasing the error tolerance.'.format(
                    max_pvalue_std_error, endpoint, initial_inflection_point))

        fit_xs = numpy.concatenate((numpy.linspace(min_val, data_mean, num=interp_points),
                                    numpy.linspace(data_mean, endpoint, num=interp_points)))
        fit_ys = numpy.log(compute_empirical_pvalue(data, values=fit_xs, tail='right', is_sorted=True))

        if initial_inflection_point == None:
            initial_inflection_point = data_mean

        p, e = scipy.optimize.curve_fit(cls._piecewise_logsf, fit_xs, fit_ys,
                                        p0=numpy.array([initial_inflection_point, initial_slope]))
        return p

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.slope)


class PiecewiseApproxLinearDirect:
    """
    Stub class for an empirical distribution with methods to:
        1. Fit a piecewise linear function to the log-survival function of a data sample
        2. Compute the value of the log-survival function for given x.
    """

    def __init__(self, inflection_point, slope):
        self.inflection_point = inflection_point
        self.slope = slope

    @staticmethod
    def _piecewise_logsf(x, inflection_point, slope):
        """
        A piecewise linear function that = 0 for all x < :param:`inflection_point`
            and rises linearly with slope :param:`slope` for all points > :param:`inflection_point`
        """
        return numpy.piecewise(x, [x < inflection_point], [lambda x: 0, lambda x: slope * (x - inflection_point)])

    @classmethod
    def fit(cls, data, is_sorted=False, max_pvalue_std_error=0.05):
        min_val, max_val = data.min(), data.max()
        data_mean = data.mean()
        endpoint = compute_empirical_quantile(data, 1 - determine_p_cutoff(n=len(data)), is_sorted=True)
        inflection_point = data_mean
        print('Max examined data point: {}'.format(endpoint))

        fit_xs = [inflection_point, endpoint]
        fit_ys = [0, numpy.log(compute_empirical_pvalue(data, values=endpoint, tail='right', is_sorted=is_sorted))]

        slope = (fit_ys[1] - fit_ys[0]) / (fit_xs[1] - fit_xs[0])

        return inflection_point, slope

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.slope)


class PiecewiseApproxPower:
    """
    Stub class for an empirical distribution with methods to:
        1. Fit a power function to the log-survival function of a data sample
        2. Compute the value of the log-survival function for given x.
    """

    def __init__(self, inflection_point, power, scale):
        self.inflection_point = inflection_point
        self.power = power
        self.scale = scale

    @staticmethod
    def _piecewise_logsf(x, inflection_point, power, scale):
        """
        A piecewise power function:
            x < :param:`inflection_point`: 0
            x >= :param:`inflection_point`: :param:`scale` * (:param:`x` - :param:`inflection_point`)**:param:`power`
        """
        assert inflection_point < x[-1], 'Inflection point must be smaller than largest x value to avoid all zeros'
        return numpy.piecewise(x, [x < inflection_point],
                               [lambda x: 0, lambda x: scale * (x - inflection_point) ** power])

    @classmethod
    def fit(cls, data, is_sorted=False, max_pvalue_std_error=0.05, interp_points=50,
            initial_inflection_point=None, initial_power=1, optimization_kwargs={}):

        if not is_sorted:
            data = numpy.sort(data)

        min_val, max_val = data.min(), data.max()
        data_mean = data.mean()
        endpoint = compute_empirical_quantile(data, 1 - determine_p_cutoff(n=len(data)), is_sorted=True)

        if endpoint <= data_mean:
            raise ValueError(
                'Minimum data value that meets target p-value error threshold of {} is {}, which is below the mean of {}, therefore piecewise linear approximation will not work. Try generating more empirical samples or increasing the error tolerance.'.format(
                    max_pvalue_std_error, endpoint, initial_inflection_point))

        fit_xs = numpy.concatenate((numpy.linspace(min_val, data_mean, num=interp_points),
                                    numpy.linspace(data_mean, endpoint, num=interp_points)))
        fit_ys = numpy.log(compute_empirical_pvalue(data, values=fit_xs, tail='right', is_sorted=True))

        if initial_inflection_point is None:
            initial_inflection_point = data_mean

        res = scipy.optimize.minimize(fun=cls._generate_obj_func(fit_xs, fit_ys),
                                      x0=numpy.array([initial_inflection_point, initial_power]),
                                      bounds=((-numpy.inf, endpoint - 1e-4),
                                              (1, numpy.inf)),
                                      method='L-BFGS-B', **optimization_kwargs
                                      )
        inflection_point, power = res.x

        first_pass_ys = cls._piecewise_logsf(fit_xs, inflection_point, power, scale=-1)
        scale = - (fit_ys[-1] / first_pass_ys[-1])

        return inflection_point, power, scale

    @classmethod
    def _generate_obj_func(cls, fit_xs, fit_ys):
        def obj_func(params):
            inflection_point, power = params
            test_ys = cls._piecewise_logsf(x=fit_xs, inflection_point=inflection_point, power=power, scale=-1)

            score = scipy.stats.pearsonr(test_ys, fit_ys)[0]
            return -score

        return obj_func

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.power, self.scale)

import numpy
import scipy.stats
from empdist.empirical_pval import empirical_p_val
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
    def fit(cls, data, is_sorted=False, support_quantile=DEFAULT_SUPPORT_QUANTILE, interp_points=50,
            initial_inflection_point=None, initial_slope=500):
        if not is_sorted:
            data = numpy.sort(data)

        data_min, data_max = data[0], data[-1]
        extent = data_max - data_min
        midpoint = (data_min + data_max) / 2

        support = midpoint - extent, midpoint + (extent / 2) * support_quantile

        if initial_inflection_point == None:
            initial_inflection_point = data.mean()
        # print('min: {} max: {} midpoint: {} extent: {} support: {}'.format(data_min, data_max, midpoint, extent, support))

        fit_xs = numpy.linspace(*support, num=interp_points)
        fit_ys = numpy.log(empirical_p_val(data, values=fit_xs, tail='right', is_sorted=True))
        p, e = scipy.optimize.curve_fit(cls._piecewise_logsf, fit_xs, fit_ys,
                                        p0=[initial_inflection_point, initial_slope])
        return p

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
        return numpy.piecewise(x, [x < inflection_point],
                               [lambda x: 0, lambda x: scale * (x - inflection_point) ** power])

    @classmethod
    def fit(cls, data, is_sorted=False, support_quantile=DEFAULT_SUPPORT_QUANTILE, interp_points=50,
            initial_inflection_point=None, initial_power=2, initial_scale=-1e6):
        if not is_sorted:
            data = numpy.sort(data)

        data_min, data_max = data[0], data[-1]
        extent = data_max - data_min
        midpoint = (data_min + data_max) / 2

        support = midpoint - extent / 2, midpoint + (extent * support_quantile / 2)

        if initial_inflection_point == None:
            initial_inflection_point = data.mean()
        # print('min: {} max: {} midpoint: {} extent: {} support: {}'.format(data_min, data_max, midpoint, extent, support))

        fit_xs = numpy.linspace(*support, num=interp_points)
        fit_ys = numpy.log(empirical_p_val(data, values=fit_xs, tail='right', is_sorted=True))
        p, e = scipy.optimize.curve_fit(cls._piecewise_logsf, fit_xs, fit_ys,
                                        p0=[initial_inflection_point, initial_power, initial_scale])
        return p

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.power, self.scale)

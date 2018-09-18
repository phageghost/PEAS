import numpy
import scipy.stats
from empdist.empirical_distributions import EmpiricalDistribution
from empdist.empirical_pval import compute_empirical_pvalue, compute_p_confidence, compute_empirical_quantile
from scipy.optimize import curve_fit

from . import constants


def rms_error(X, Y):
    return numpy.sqrt(numpy.mean((X - Y) ** 2))


def cosine_sim(X, Y):
    return numpy.dot(X, Y) / numpy.linalg.norm(X) / numpy.linalg.norm(Y)


class PiecewiseEmpiricalApprox:
    @classmethod
    def _compute_empirical_logsf(cls, data, support_range=None, unique_samples=0, max_confident_x=None,
                                 max_pvalue_cv=constants.DEFAULT_PVALUE_CV,
                                 interp_points=constants.DEFAULT_NUM_FIT_POINTS, is_sorted=False):
        if not is_sorted:
            data = numpy.sort(data)

        if not unique_samples:
            unique_samples = len(data)

        if max_confident_x is None:
            max_confident_x = compute_empirical_quantile(data, 1 - compute_p_confidence(n=unique_samples,
                                                                                        pvalue_cv=max_pvalue_cv),
                                                         is_sorted=True)

        if not support_range:
            x_min = data.min()
            x_max = min(data.max(), max_confident_x)
        else:
            x_min = max(data.min(), support_range[0])
            x_max = min(data.max(), support_range[1], max_confident_x)

        fit_xs = numpy.linspace(x_min, x_max, num=interp_points)
        fit_ys = numpy.log(compute_empirical_pvalue(data, values=fit_xs, tail='right', is_sorted=True))

        return fit_xs, fit_ys

    def sf(self, x):
        return numpy.exp(self.logsf(x))

    def cdf(self, x):
        return 1 - self.sf(x)


class PiecewiseApproxLinear(PiecewiseEmpiricalApprox):
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
    def fit_with_existing_empirical_logsf(cls, fit_xs, fit_ys, x0, optimization_kwargs={}):
        p, e = scipy.optimize.curve_fit(cls._piecewise_logsf, fit_xs, fit_ys,
                                        p0=x0, **optimization_kwargs)
        return p

    @classmethod
    def fit(cls, data, support_range=None, max_confident_x=None, is_sorted=False,
            max_pvalue_cv=constants.DEFAULT_PVALUE_CV,
            interp_points=constants.DEFAULT_NUM_FIT_POINTS,
            initial_inflection_point=None, initial_slope=500):
        fit_xs, fit_ys = cls._compute_empirical_logsf(data=data, support_range=support_range,
                                                      max_confident_x=max_confident_x,
                                                      max_pvalue_cv=max_pvalue_cv, interp_points=interp_points,
                                                      is_sorted=is_sorted)

        if initial_inflection_point is None:
            initial_inflection_point = data.mean()

        return cls.fit_with_existing_empirical_logsf(fit_xs, fit_ys, x0=(initial_inflection_point, initial_slope))

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.slope)


class PiecewiseApproxLinearDirect(PiecewiseEmpiricalApprox):
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
    def fit(cls, data, is_sorted=False):
        data_mean = data.mean()
        endpoint = compute_empirical_quantile(data, 1 - compute_p_confidence(n=len(data)), is_sorted=True)
        inflection_point = data_mean

        fit_xs = [inflection_point, endpoint]
        fit_ys = [0, numpy.log(compute_empirical_pvalue(data, values=endpoint, tail='right', is_sorted=is_sorted))]

        slope = (fit_ys[1] - fit_ys[0]) / (fit_xs[1] - fit_xs[0])

        return inflection_point, slope

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.slope)


class PiecewiseApproxPower(PiecewiseEmpiricalApprox):
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
        # assert inflection_point < x[-1], 'Inflection point must be smaller than largest x value to avoid all zeros'
        return numpy.piecewise(x, [x < inflection_point],
                               [lambda x: 0, lambda x: scale * (x - inflection_point) ** power])

    @classmethod
    def fit_with_existing_empirical_logsf(cls, fit_xs, fit_ys, x0=(None, 1.2), optimization_kwargs={}):
        initial_inflection_point, initial_power = x0

        res = scipy.optimize.basinhopping(func=cls._generate_obj_func(fit_xs, fit_ys),
                                          x0=numpy.array([initial_inflection_point, initial_power]),
                                          minimizer_kwargs={'bounds': ((-numpy.inf, fit_xs[-1] - 1e-6),
                                                                       (1, numpy.inf)),
                                                            'method': 'L-BFGS-B'},
                                          **optimization_kwargs
                                          )
        inflection_point, power = res.x

        first_pass_ys = cls._piecewise_logsf(fit_xs, inflection_point, power, scale=-1)
        scale = -(fit_ys.mean() / first_pass_ys.mean())

        return inflection_point, power, scale

    @classmethod
    def fit(cls, data, support_range=None, max_confident_x=None, is_sorted=False, unique_samples=0,
            max_pvalue_cv=constants.DEFAULT_PVALUE_CV, interp_points=constants.DEFAULT_NUM_FIT_POINTS,
            x0=(None, 1.2), optimization_kwargs={}):

        initial_inflection_point, initial_power = x0
        fit_xs, fit_ys = cls._compute_empirical_logsf(data, support_range=support_range,
                                                      max_confident_x=max_confident_x, unique_samples=unique_samples,
                                                      max_pvalue_cv=max_pvalue_cv,
                                                      interp_points=interp_points, is_sorted=is_sorted)

        if initial_inflection_point is None:
            initial_inflection_point = data.mean()

        return cls.fit_with_existing_empirical_logsf(fit_xs, fit_ys, x0=(initial_inflection_point, initial_power),
                                                     optimization_kwargs=optimization_kwargs)

    @classmethod
    def _generate_obj_func(cls, fit_xs, fit_ys):
        def obj_func(params):
            inflection_point, power = params
            test_ys = cls._piecewise_logsf(x=fit_xs, inflection_point=inflection_point, power=power, scale=-1)

            score = -cosine_sim(fit_ys, test_ys)

            if numpy.isnan(score) or numpy.isinf(score):
                return numpy.inf

            return score

        return obj_func

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.power, self.scale)


class PiecewiseApproxPowerSum(PiecewiseEmpiricalApprox):
    """
    Stub class for an empirical distribution with methods to:
        1. Fit a power function to the log-survival function of a data sample
        2. Compute the value of the log-survival function for given x.
    """

    def __init__(self, inflection_point, power_a, power_b, scale):
        self.inflection_point = inflection_point
        self.power_a = power_a
        self.power_b = power_b
        self.scale = scale

    @staticmethod
    def _piecewise_logsf(x, inflection_point, power_a, power_b, scale):
        """
        A polynomial function:
            x < :param:`inflection_point`: 0
            x >= :param:`inflection_point`: :param:`scale` * ((:param:`x` - :param:`inflection_point`)**:param:`power_a` + :param:`scale` * ((:param:`x` - :param:`inflection_point`)**:param:`power_b`
        """
        # assert inflection_point < x[-1], 'Inflection point must be smaller than largest x value to avoid all zeros'
        return numpy.piecewise(x, [x < inflection_point],
                               [lambda x: 0, lambda x: scale * (
                                       (x - inflection_point) ** power_a + (x - inflection_point) ** power_b)])

    @classmethod
    def fit_with_existing_empirical_logsf(cls, fit_xs, fit_ys, x0=(None, 1.5, 1.1), optimization_kwargs={}):
        initial_inflection_point, initial_power_a, initial_power_b = x0

        res = scipy.optimize.basinhopping(func=cls._generate_obj_func(fit_xs, fit_ys),
                                          x0=numpy.array([initial_inflection_point, initial_power_a, initial_power_b]),
                                          minimizer_kwargs={'bounds': ((-numpy.inf, fit_xs[-1] - 1e-6),
                                                                       (1, numpy.inf),
                                                                       (1, numpy.inf)),
                                                            'method': 'L-BFGS-B'},
                                          **optimization_kwargs
                                          )
        inflection_point, power_a, power_b = res.x

        first_pass_ys = cls._piecewise_logsf(fit_xs, inflection_point, power_a, power_b, scale=-1)
        scale = -(fit_ys.mean() / first_pass_ys.mean())
        
        power_a, power_b = sorted(power_a, power_b) # force ordering

        return inflection_point, power_a, power_b, scale

    @classmethod
    def fit(cls, data, support_range=None, max_confident_x=None, unique_samples=0, is_sorted=False,
            max_pvalue_cv=constants.DEFAULT_PVALUE_CV, interp_points=constants.DEFAULT_NUM_FIT_POINTS,
            x0=(None, 1.5, 1.1), optimization_kwargs={}):

        initial_inflection_point, initial_power_a, initial_power_b = x0

        fit_xs, fit_ys = cls._compute_empirical_logsf(data, support_range=support_range,
                                                      max_confident_x=max_confident_x, unique_samples=unique_samples,
                                                      max_pvalue_cv=max_pvalue_cv,
                                                      interp_points=interp_points, is_sorted=is_sorted)

        if initial_inflection_point is None:
            initial_inflection_point = data.mean()

        return cls.fit_with_existing_empirical_logsf(fit_xs, fit_ys,
                                                     x0=(initial_inflection_point, initial_power_a, initial_power_b),
                                                     optimization_kwargs=optimization_kwargs)

    @classmethod
    def _generate_obj_func(cls, fit_xs, fit_ys):
        def obj_func(params):
            inflection_point, power_a, power_b = params
            test_ys = cls._piecewise_logsf(x=fit_xs, inflection_point=inflection_point, power_a=power_a,
                                           power_b=power_b, scale=-1)

            score = -cosine_sim(fit_ys, test_ys)

            if numpy.isnan(score) or numpy.isinf(score):
                return numpy.inf

            return score

        return obj_func

    def logsf(self, x):
        return self._piecewise_logsf(x, self.inflection_point, self.power_a, self.power_b, self.scale)


class HybridDistribution(PiecewiseEmpiricalApprox):
    """
    """

    def __init__(self, empirical_distribution, extrapolated_distribution, crossover_point):
        self.crossover_point = crossover_point
        self.empirical_distribution = empirical_distribution
        self.extrapolated_distribution = extrapolated_distribution

    @classmethod
    def fit(cls, data, support_range=None, unique_samples=0, is_sorted=False, max_pvalue_cv=constants.DEFAULT_PVALUE_CV,
            interp_points=constants.DEFAULT_NUM_FIT_POINTS,
            extrapolated_distribution_class=PiecewiseApproxPowerSum, optimization_kwargs={}):
        """
        Basic idea is to fit a power law tail to the portion of the data that falls between the
        data min and the max confident value, and then that max confident value becomes our 
        crossover point. We will generate a histogram-based EmpiricalDistribution for that same
        data range. Logsf queries below the crossover point will be answered from the EmpiricalDistribution,
        and queries above the crossover point will come from the power law tail.
        
        Forsee difficulty in getting a smooth transition . . . 
        
        """
        if not is_sorted:
            data = numpy.sort(data)

        if not unique_samples:
            unique_samples = len(data)

        max_confident_x = compute_empirical_quantile(data, 1 - compute_p_confidence(n=unique_samples,
                                                                                    pvalue_cv=max_pvalue_cv),
                                                     is_sorted=True)

        empirical_distribution = EmpiricalDistribution.from_data(data)

        extrapolated_distribution = extrapolated_distribution_class(*extrapolated_distribution_class.fit(data=data,
                                                                                                         max_confident_x=max_confident_x,
                                                                                                         support_range=support_range,
                                                                                                         unique_samples=unique_samples,
                                                                                                         max_pvalue_cv=max_pvalue_cv,
                                                                                                         is_sorted=True,
                                                                                                         interp_points=interp_points,
                                                                                                         optimization_kwargs=optimization_kwargs))

        return empirical_distribution, extrapolated_distribution, max_confident_x

    def logsf(self, x):
        use_empirical = x <= self.crossover_point
        empirical_xs = x[use_empirical]
        empirical_ys = self.empirical_distribution.logsf(empirical_xs)

        extrapolated_xs = x[numpy.logical_not(use_empirical)]
        extrapolated_ys = self.extrapolated_distribution.logsf(extrapolated_xs)

        return numpy.concatenate((empirical_ys, extrapolated_ys))

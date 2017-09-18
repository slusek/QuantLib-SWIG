"""
 Copyright (C) 2017 Wojciech Ślusarski

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
"""

import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('blackDeltaVolSurfaceExample')

def blackFormulaSpotDelta(strike, spot, forward, volatility, expiryT,
                          discount=1.0, displacement=0.0):
    """
    Calculates spot delta for call option, does not handle premium included
    delta
    :param strike:
    :param spot:
    :param forward:
    :param volatility:
    :param expiryT:
    :param discount:
    :param displacement:
    :return:
    """
    df_q = forward / spot * discount
    forward += displacement
    strike += displacement
    d_denom = (volatility * expiryT ** 0.5)
    d_1 = np.log(forward / strike) + 0.5 * volatility * volatility * expiryT
    d_1 /= d_denom
    N = ql.CumulativeNormalDistribution()
    delta = df_q * N(d_1)

    return delta


class BlackDeltaVolSurface(ql.BlackVolTermStructure):

    #Precision required in Newton-Raphson algorithm to determine volatility
    # level
    VOL_PRECISION = 1e-6

    def __init__(self,
                 reference_date,
                 calendar,
                 deltas,
                 expiry_tenors,
                 black_delta_vol_matrix,
                 day_counter,
                 spot,
                 dom_df,
                 for_df,
                 interpolator=ql.MonotonicCubicNaturalSpline,
                 lower_extrapolator=None,
                 upper_extrapolator=None
                 ):

        """

        :param reference_date:
        :param calendar:
        :param deltas:
        :param expiry_tenors:
        :param black_delta_vol_matrix: delta levels in rows, expiry time
        nodes in columns
        :param day_counter:
        :param interpolator:
        :param lower_extrapolator:
        :param upper_extrapolator:
        """
        ql.BlackVolTermStructure.__init__(self)

        self.reference_date = reference_date
        self.calendar = calendar
        self.deltas = deltas / 100.0

        self._check_deltas_sorted()

        self.expiry_tenors = expiry_tenors
        self._black_delta_vol_matrix = black_delta_vol_matrix
        self.day_counter = day_counter
        self.spot = spot
        self.dom_df = dom_df
        self.for_df = for_df
        self.interpolator = interpolator
        #not yet handled
        #self.lower_extrapolator = lower_extrapolator
        #self.upper_extrapolator = upper_extrapolator

        self.spot_date = None
        self._expiry_dates = None
        self._expiry_time = None

        self.time_interpolator = ql.LinearInterpolation

        variance = np.power(self._black_delta_vol_matrix, 2)
        self._variance = np.multiply(self.expiry_time, variance.T)


    def _check_deltas_sorted(self):
        for i in range(len(self.deltas)-1):
            assert self.deltas[i+1] > self.deltas[i], "Delta vector should be " \
                                                      "sorted in ascending order"

    @property
    def expiry_dates(self):
        if self._expiry_dates is None:
            spot_date = self.calendar.advance(self.reference_date,
                                              ql.Period('2d'),
                                              ql.Following)
            self.spot_date = ql.UnitedStates().adjust(spot_date,
                                                      ql.Following)
            self._expiry_dates = [self._expiry_from_period(ql.Period(period))
                                  for period in self.expiry_tenors]
        return self._expiry_dates

    def _expiry_from_period(self, period):
        if period < ql.Period('1M'):
            date = self.calendar.advance(self.reference_date,
                                        period,
                                        ql.Following)
            date = ql.UnitedStates().adjust(date,
                                            ql.Following)
        else:
            delivery = self.calendar.advance(self.spot_date,
                                            period,
                                            ql.Following)
            delivery = ql.UnitedStates().adjust(delivery,
                                                ql.Following)
            date = self.calendar.advance(delivery, -2, ql.Days, ql.Preceding)
            date = ql.UnitedStates().adjust(date,
                                            ql.Preceding)
        return date

    @property
    def expiry_time(self):
        if self._expiry_time is None:
            self._expiry_time = [self.day_counter.yearFraction(
                                                    self.reference_date,
                                                    exp_date)
                                 for exp_date in self.expiry_dates]
        return self._expiry_time



    def smile(self, time):
        """
        Returns volatility velvel for each delta node interpolated for a
        specified expiry `time`
        :param time: expiry time for which volatility is interpolated
        :return: volatility levels at each delta node (kind of interpolated
        smile section)
        """
        smile = [self._time_intepolation(time,
                                         self.expiry_time,
                                         variance_vector.tolist())
                 for variance_vector in self._variance]
        return smile

    def _time_intepolation(self, T, time_nodes, delta_variance_nodes):
        """
        Returns volatility interpolated for a given expiry time t from a
        variance vector at a given delta level.
        In QuantLib probably BlackVarianceCurve could be utilized here,
        we need to have some method for interpolation of variance at a given
        time delta level, consistent for each delta.
        :param T: expiry time for which volatility should be determined
        :param time_nodes: vector of time nodes for which variance is available
        :param delta_variance_nodes: vector of variances (:math:`\sigma^2T`)
            which are interpolated
        :return: volatility :math:`\sigma_T` for the expiry :math:`T`
        :rtype: float
        """
        interp = self.time_interpolator(time_nodes, delta_variance_nodes)
        vol = (interp(T, True) / T) ** 0.5
        return vol


    def blackVol(self, strike, time):
        logger.debug("Interpolating vol for:\n"
                     "strike: {:.4f}\n"
                     "expiry: {:.3f}\n".format(strike, time))
        smile = self.smile(time)
        # 1. determine strike for each delta node
        strikes = [ql.BlackDeltaCalculator(ql.Option.Call,
                                           ql.DeltaVolQuote.Spot,
                                           self.spot.value(),
                                           self.dom_df.discount(time),
                                           self.for_df.discount(time),
                                           vol*time**0.5)
                            .strikeFromDelta(delta_level)
                   for delta_level, vol in zip(self.deltas, smile)]
        # 2. get initial vol ATM is one choice, I prefer well interpolated
        # already, that is why first I determined strikes for each delta.
        logger.debug("Strikes: {}".format(strikes))
        logger.debug("Vols: {}".format(smile))
        # Quantlib Interpolator requires ascending sorting
        idx = list(np.argsort(strikes))
        strike_interpolator = ql.MonotonicCubicNaturalSpline(
                                                list(np.array(strikes)[idx]),
                                                list(np.array(smile)[idx]))
        try:
            vol = strike_interpolator(strike, False)
        except RuntimeError as err:
            logger.debug("Flat extrapolation")
            # not sure how to properly provide flat extrapolation
            if strike < np.min(strikes):
                return smile[-1]
            else:
                return smile[0]
        # 3. We have pretty good initial vol, now it can be interpolated in
        # delta space
        #payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        dom_df = self.dom_df.discount(time)
        forward = self.spot.value() * self.for_df.discount(time) / dom_df

        # Now it would be nice, to have an object calculating delta with
        # a setter method for updating the volatility or accepting
        # volatility as a QuoteHandle. Since I haven't found yet such object
        # I will use functions defined above
        delta = blackFormulaSpotDelta(strike,
                                      self.spot.value(),
                                      forward,
                                      vol,
                                      time,
                                      dom_df,
                                      0.0)
        idx = list(np.argsort(self.deltas))
        logger.debug(self.deltas)
        delta_interpolator = ql.MonotonicCubicNaturalSpline(list(self.deltas),
                                                            smile)

        try:
            # not sure how to impose flat extrapolation in Python
            new_vol = delta_interpolator(delta, False)
        except RuntimeError:
            if delta < self.deltas[0]:
                new_vol = smile[0]
            else:
                new_vol = smile[-1]

        iternum = 0
        logger.debug("Iteration: {}, delta: {:,.6f} "
                     "vol: {:,.6f}".format(iternum,
                                           delta,
                                           new_vol))
        while (abs(new_vol - vol) > BlackDeltaVolSurface.VOL_PRECISION):
            iternum += 1
            logger.debug("Iteration: {}, delta: {:,.6f} "
                         "vol: {:,.6f}".format(iternum,
                                               delta,
                                               new_vol))
            delta = blackFormulaSpotDelta(strike,
                                          self.spot.value(),
                                          forward,
                                          new_vol,
                                          time,
                                          dom_df,
                                          0.0)

            vol = new_vol

            try:
                # not sure how to impose flat extrapolation
                new_vol = delta_interpolator(delta, False)
            except RuntimeError:
                if delta < self.deltas[0]:
                    new_vol = smile[0]
                else:
                    new_vol = smile[-1]

            if iternum > 100:
                logger.warning("Reached maximum iterations without convergence")
                return new_vol

        return new_vol


        return vol

def main():
    todaysDate = ql.Date(5, ql.September, 2017)
    ql.Settings.instance().evaluationDate = todaysDate
    spotDate = ql.Date(7, ql.September, 2017)
    domestic_rate = ql.FlatForward(spotDate, 0.017, ql.Actual365Fixed())
    foreign_rate = ql.FlatForward(spotDate, 0.013, ql.Actual365Fixed())

    # market data
    underlying = ql.SimpleQuote(3.7)

    calendar = ql.JointCalendar(ql.Poland(), ql.UnitedStates())

    # delta levels
    deltas = [d for d in range(10, 100, 10)]

    # some hypothetical volatility levels on each delta point, determined from
    # some artificial polynomial
    vols = [3.32e-5 * d ** 2 - 0.0027 * d + 0.1776
            for d in deltas]
    expiry_periods = ['1W', '1M', '3M', '6M', '1Y']
    vols = np.tile(np.array(vols), (len(expiry_periods), 1))

    bdv = BlackDeltaVolSurface(todaysDate,
                               calendar,
                               np.array(deltas),
                               expiry_periods,
                               np.array(vols),
                               ql.Actual365Fixed(),
                               underlying,
                               domestic_rate,
                               foreign_rate)

    strikes_num = 50
    expiry_num = 20
    strikes = np.linspace(2.5, 5.0, strikes_num)
    expiry = np.linspace(1/52, 36/24, expiry_num)
    strikes, expiry = np.meshgrid(strikes, expiry)
    interp_vols = [bdv.blackVol(strike, time)
                   for strike, time in zip(strikes.flatten(),
                                           expiry.flatten())]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_trisurf(expiry.flatten(),
                    strikes.flatten(),
                    interp_vols,
                    linewidth=0, antialiased=True,
                    cmap='viridis')
    ax.set_xlabel('Time to expiry')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Volatility')
    ax.view_init(20, 15)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    main()

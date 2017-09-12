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


def blackFormulaVanna(strike, spot, forward, volatility, expiryT,
                      discount=1.0, displacement=0.0):
    """
    Returns Vanna of a European Option according to Black76 formula.
    Alternative name for this derivative is DdeltaDvol.
    It has the same value for call and put option.

    :param optionType: QuantLibs payof
    :param strike: options' strike
    :param forward: underlying forward price
    :param volatility: annualized volatility
    :param expiryT: time to expiry
    :param discount: discount factor
    :param displacement: displacement parameter `a` in displaced diffusion model
        :math:`dF_t = \sigma \left(F_t + a \right) dW_t`
    :return: vanna value
    :rtype: float
    """
    # dividend/ foreign currency / convenience yield discount factor
    df_q = forward / spot / discount
    forward += displacement
    strike += displacement
    d_denom = (volatility * expiryT ** 0.5)
    d_1 = np.log(forward / strike) + 0.5 * volatility * volatility * expiryT
    d_1 /= d_denom
    d_2 = d_1 - d_denom

    vanna = df_q * d_2 / volatility * ql.NormalDistribution().derivative(d_1)

    return vanna

class BlackDeltaVolSurface(ql.BlackVolTermStructure):
    def __init__(self,
                 reference_date,
                 calendar,
                 deltas,
                 expiry_dates,
                 black_delta_vol_matrix,
                 day_counter,
                 lower_extrapolator,
                 upper_extrapolator):
        ql.BlackVolTermStructure.__init__(self)



def main():
    x = blackFormulaVanna()
    todaysDate = ql.Date(5, ql.September, 2017)
    ql.Settings.instance().evaluationDate = todaysDate
    spotDate = ql.Date(7, ql.September, 2017)
    domestic_rate = ql.FlatForward(spotDate, 0.017, ql.Actual365Fixed())
    foreign_rate = ql.FlatForward(spotDate, 0.013, ql.Actual365Fixed())
    # option parameters
    exercise = ql.EuropeanExercise(ql.Date(22, ql.November, 2017))
    deliveryDate = ql.Date(24, ql.November, 2017)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, 3.9)

    # market data
    underlying = ql.SimpleQuote(3.7)

    # delta levels
    deltas = [d for d in range(10, 100, 10)]

    # some hypothetical volatility levels on each delta point, determined from
    # some artificial polynomial
    vols = [3.32e-5 * d ** 2 - 0.0027 * d + 0.1776
            for d in deltas]
    time_to_vol_expiry = 0.25

    calendar = ql.JointCalendar(ql.Poland(), ql.UnitedStates())

    volatility_quotes = [ql.DeltaVolQuote(delta,
                                          ql.QuoteHandle(ql.SimpleQuote(vol)),
                                          time_to_vol_expiry,
                                          ql.DeltaVolQuote.Spot)
                         for vol, delta in zip(vols, deltas)]

    constant_volatility = ql.BlackConstantVol(todaysDate, ql.TARGET(), 0.10,
                                              ql.Actual365Fixed())

    process = ql.BlackScholesMertonProcess(ql.QuoteHandle(underlying),
                                           ql.RelinkableYieldTermStructureHandle(
                                               foreign_rate),
                                           ql.RelinkableYieldTermStructureHandle(
                                               domestic_rate),
                                           ql.BlackVolTermStructureHandle(
                                               constant_volatility))

    option = ql.VanillaOption(payoff, exercise)

    # method: analytic
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    value = option.NPV()
    print("Analytical option price: {:,.2f}".format(value))

    bdv = BlackDeltaVolSurface(todaysDate,
                               calendar,
                               [0.1, 0.2, 0.3],
                               [ql.Date(1, 10, 2017), ql.Date(1, 12, 2017)],
                               [[0.1, 0.1], [0.2, 0.2], [0.25, 0.25]],
                               ql.Actual365Fixed(),
                               ql.BackwardFlat(),
                               ql.ForwardFlat())
    print(bdv.dayCounter())


if __name__ == '__main__':
    main()

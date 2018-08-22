
# Copyright (C) 2004, 2005, 2006, 2007, 2018 StatPro Italia srl, Wojciech Ślusarski
#
# This file is part of QuantLib, a free-software/open-source library
# for financial quantitative analysts and developers - http://quantlib.org/
#
# QuantLib is free software: you can redistribute it and/or modify it under the
# terms of the QuantLib license.  You should have received a copy of the
# license along with this program; if not, please email
# <quantlib-dev@lists.sf.net>. The license is also available online at
# <http://quantlib.org/license.shtml>.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the license for more details.

from QuantLib import *
from math import exp

# global data
todaysDate = Date(23, August, 2018)
Settings.instance().evaluationDate = todaysDate
settlementDate = Date(25, August, 2018)
riskFreeRate = FlatForward(todaysDate, 0.02, Actual365Fixed())

# market data
underlying = SimpleQuote(4.3)
volatility = BlackConstantVol(todaysDate, TARGET(), 0.06, Actual365Fixed())
dividendYield = FlatForward(todaysDate, 0.005, Actual365Fixed())

# option parameters
expiry_date = Date(26, August, 2019)
time_to_expiry = (expiry_date - todaysDate) / 365
exercise = EuropeanExercise(expiry_date)
payoff_amount = 1.0
# 0-delta straddle strike
strike = underlying.value() * exp((0.02 - 0.005 - 0.5 * 0.06 ** 2) * time_to_expiry) 
payoff = CashOrNothingPayoff(Option.Call, strike, payoff_amount)


# report
header = ' |'.join(['%17s' % tag for tag in ['method','value',
                                            'estimated error',
                                            'actual error' ] ])
print('')
print(header)
print('-'*len(header))

refValue = None
def report(method, x, dx = None):
    e = '%.4f' % abs(x-refValue)
    x = '%.5f' % x
    if dx:
        dx = '%.4f' % dx
    else:
        dx = 'n/a'
    print(' |'.join(['%17s' % y for y in [method, x, dx, e] ]))


# good to go

process = BlackScholesMertonProcess(QuoteHandle(underlying),
                                    YieldTermStructureHandle(dividendYield),
                                    YieldTermStructureHandle(riskFreeRate),
                                    BlackVolTermStructureHandle(volatility))

option = VanillaOption(payoff, exercise)

# method: analytic
option.setPricingEngine(AnalyticEuropeanEngine(process))
value = option.NPV()
refValue = value
report('analytic',value)

# method: integral
option.setPricingEngine(IntegralEngine(process))
report('integral',option.NPV())

# method: finite differences
timeSteps = 801
gridPoints = 800

option.setPricingEngine(FDEuropeanEngine(process,timeSteps,gridPoints))
report('finite diff.',option.NPV())

expected = 0.5 * exp(-0.02 * time_to_expiry)
print("Value expected was: {:,.5}".format(expected))
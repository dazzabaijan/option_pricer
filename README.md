# Quantitative Finance Projects
This repo contains all my quantitative finance related projects that I'm implementing in my free time.

##Option Pricers

An option pricer library that encapsulates all the numerical option pricers that I will be implementing

Done:
  - Cox-Ross-Rubinstein market model (Binomial Model) for pricing European and American options

To do:
  - Leisen-Reimer tree
  - Trinomial trees in option pricing
  - Explicit and Implicit finite differences method for pricing American and European options
  - Crank-Nicholson method of finite differences method for pricing European option
  - Longstaff-Schwartz method for pricing American options
  
##Kalman Filter

A Kalman filter to dynamically calculate the time-varying slope and intercept between two assets.
To do:
  - Implement a basic version of a Kalman filter from scratch using Numpy and Matplotlib. The rough idea is to take two closely related assets and perform linear regressions on them over time to dynamically estimate the gradient and intercept so as to determine how much of each asset to long and short at a particular threshold and hence determine the hedging ratio between the two assets, where one of them will be the "observed" variables.

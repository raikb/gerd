[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://github.com/raikb/gerd/workflows/Python%203.7/badge.svg)
![Python 3.8](https://github.com/raikb/gerd/workflows/Python%203.8/badge.svg)

# gerd

> Essentially, all models are wrong, but some are useful.
> - George E. P. Box

Hopefully, gerd can be of use for someone.

## What is gerd?

**gerd.Dispatch** is an easy-to-use multi-area power market model that applies mixed-integer programming (MIP) and relaxed MIP (RMIP) for solving.

### **gerd.Dispatch** allows you to model:
* Power prices
* Power plant dispatch:
  - Minimum up-time of generators
  - Minimum down-time of generators
* Storages (batteries, pump storages etc.)
* Must-run production
* Cross-border flows

### Main features:
* Relies on the google package [ortools](https://developers.google.com/optimization):
  -  Fast, memory efficient and numerically stable solvers that do not require any cumbersome third-party installations
* Input-driven modeling to reduce needless parameter settings:
  - The input handed over to the model defines what is modeled
  - Options to switch on and off model features still exist
* Rolling horizon optimization for cutting down run-time can be chosen
* Different solving options available:
  - MIP if the dispatch is important
  - RMIP if prices are important and if computational speed counts
  - Combination of MIP and RMIP if the dispatch and the prices are important

## How to use it?

### Installation
```
pip install gerd
```
### Read input from CSV or use pandas DataFrames directly
Example input files can be found in [examples](examples/).
```python
from gerd import models
import pandas as pd

# Generators
input_gens = pd.read_csv('input_generators.csv', index_col='name')
# Load
input_load = pd.read_csv(
    'input_load.csv', index_col='time', parse_dates=True)
# Variable costs
input_var_costs = pd.read_csv(
    'input_var_costs.csv', index_col='time', parse_dates=True)
```
### Define what to model and optimize
Going with the default and what is defined by the input, i.e. the minimum up-time of generators is modeled if it is defined in the input data.
```python
my_model = models.Dispatch(input_data)
my_model.optimize()
```
### Have a look at the results
```python
my_model.solution['prices'].plot()
```

## Current limitations
* The rolling horizon optimization works currently only for hourly input data and the index needs to be a pandas DatetimeIndex.

## What is next?
Possible extension of **gerd.Dispatch**:
* Definition of dynamic power plant (un)availabilities
* Modeling of spinning reserves
* Advanced time series input checking

## Acknowledgments
* Unit-commitment equations have been inspired by:
  - Kerstin Dächert and Christoph Weber: Linear reformulations of the unit commitment problem, OR 2016 Hamburg, 1.9.2016
* Many ideas, especially the rolling horizon optimization implementation, are taking from:
  - [Chi Kong Chyong, David Newbery and Thomas McCarty: A Unit Commitment and Economic Dispatch Model of the GB Electricity Market – Formulation and Application to Hydro Pumped Storage. 2019, CWPE1968](http://www.econ.cam.ac.uk/research-files/repec/cam/pdf/cwpe1968.pdf)
* Further unit-commitment equations were taken from:
  - [Van den Bergh, Kenneth Bruninx, Erik Delarue and William D‘haeseleer: LUSYM: a Unit Commitment Model formulated as a Mixed-Integer Linear Program, 2015, TME WORKING PAPER - Energy and Environment](https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wpen2014-7.pdf)

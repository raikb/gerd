'''
Example with one day of data.
'''
from gerd import models
import pandas as pd

# %% Read CSV input files
# Generators
input_gens = pd.read_csv('input_generators.csv', index_col='name')
# Storages
input_stos = pd.read_csv('input_pump_storages.csv', index_col='name')
# Load
input_load = pd.read_csv(
    'input_load_1d.csv', index_col='time', parse_dates=True)
# Variable costs
input_var_costs = pd.read_csv(
    'input_var_costs_1d.csv', index_col='time', parse_dates=True)

input_data = {'generators': input_gens,
              'storages': input_stos,
              'load': input_load,
              'var_costs': input_var_costs}

# %% Optimize without pump storage
my_model = models.Dispatch(input_data, model_storages=False)
my_model.optimize()
# Look at the results
my_model.solution['prices'].plot()
my_model.solution['prod'].plot(kind='bar', stacked=True)

# %% Optimize with pump storage
my_model = models.Dispatch(input_data)
my_model.optimize()
# Look at the results
my_model.solution['prices'].plot()
my_model.solution['prod'].plot(kind='bar', stacked=True)
my_model.solution['storage_cap'].plot()

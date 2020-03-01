'''
Example with one year of data.
'''
from gerd import models
import pandas as pd
import time

# %% Read CSV input files
# Generators
input_gens = pd.read_csv('input_generators.csv', index_col='name')
# Storages
input_stos = pd.read_csv('input_pump_storages.csv', index_col='name')
# Load
input_load = pd.read_csv(
    'input_load_1y.csv', index_col='time', parse_dates=True)
# Variable costs
input_var_costs = pd.read_csv(
    'input_var_costs_1y.csv', index_col='time', parse_dates=True)

input_data = {'generators': input_gens,
              'storages': input_stos,
              'load': input_load,
              'var_costs': input_var_costs}


# %% Optimize as MIP first and then as a RMIP without pump storage
# Should take about 2 minutes on an average machine
my_model = models.Dispatch(input_data, model_storages=False)
time_start = time.time()
my_model.optimize(rolling_optimization=True)
print(f'Took {time.time() - time_start} seconds')
# Look at the results
my_model.solution['prices'].iloc[:168].plot()
my_model.solution['prod'].iloc[:168].plot(kind='bar', stacked=True)

# %% Optimize with pump storage
# Should take about 7 minutes on an average machine
my_model = models.Dispatch(input_data)
time_start = time.time()
my_model.optimize(rolling_optimization=True)
print(f'Took {time.time() - time_start} seconds')
# Look at the results
my_model.solution['prices'].iloc[:168].plot()
my_model.solution['prod'].iloc[:168].plot(kind='bar', stacked=True)
# Charging and discharging
charging = - my_model.solution['charge'].rename(
    columns={'de_pump': 'charging'})
discharging = my_model.solution['discharge'].rename(
    columns={'de_pump': 'discharging'})
prod_incl_pump = my_model.solution['prod'].join(charging.join(discharging))
# Production including pump storage
prod_incl_pump.iloc[:168].plot(kind='bar', stacked=True)

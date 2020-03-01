import pandas as pd
import pytest

# Run for testing: pytest --cov=gerd --cov-report term-missing tests


@pytest.fixture
def dispatch_input_one_area():
    '''One area data-set. Mainly used for testing basic functions
    like prices.'''
    # Generators
    input_generators = pd.DataFrame(
        columns=['area', 'p_min', 'p_max', 'min_up_time',
                 'min_dn_time', 'startup_costs', 'p_must_run'],
        data=[['de', 300, 800, 2, 2, 0, 0],
              ['de', 100, 500, 1, 1, 0, 0]],
        index=['coal', 'gas'])
    input_generators.index.name = 'name'
    # Storages
    input_storages = pd.DataFrame(
        columns=['area', 'cap_ini', 'cap_max', 'efficiency', 'power'],
        data=[['de', 100, 200, 0.75, 100]],
        index=['pump'])
    input_storages.index.name = 'name'
    # Load
    input_load = pd.DataFrame(
        data={'de': [400, 1100, 1000, 400]},
        index=['t01', 't02', 't03', 't04'])
    input_load.index.name = 'time'
    # Variable costs
    input_var_costs = pd.DataFrame(
        data={'de_coal': [15] * 4, 'de_gas': [30] * 4},
        index=['t01', 't02', 't03', 't04'])
    input_var_costs.index.name = 'time'

    input_data = {
        'generators': input_generators,
        'storages': input_storages,
        'load': input_load,
        'var_costs': input_var_costs}

    return input_data


@pytest.fixture
def dispatch_input_one_area_long():
    '''One area data-set with longer time series. Mainly used for testing
    rolling horizon optimization.'''

    time_index = pd.date_range(
        start='2020-01-01 00:00',  periods=1000, freq='h')

    # Generators
    input_generators = pd.DataFrame(
        columns=['area', 'p_min', 'p_max', 'min_up_time',
                 'min_dn_time', 'startup_costs', 'p_must_run'],
        data=[['de', 300, 800, 2, 2, 0, 0],
              ['de', 100, 500, 1, 1, 0, 0]],
        index=['coal', 'gas'])
    input_generators.index.name = 'name'
    # Load
    input_load = pd.DataFrame(
        data={'de': [400] * 500 + [1000] * 500},
        index=time_index)
    input_load.index.name = 'time'
    # Variable costs
    input_var_costs = pd.DataFrame(
        data={'de_coal': [15] * len(time_index),
              'de_gas': [30] * len(time_index)},
        index=time_index)
    input_var_costs.index.name = 'time'

    input_data = {
        'generators': input_generators,
        'load': input_load,
        'var_costs': input_var_costs}

    return input_data


@pytest.fixture
def dispatch_input_one_area_up_time():
    '''One area data-set with longer time series and long minimum up
    times in order to further test the rolling horizon optimization
    and the modeling of minimum up time.'''

    time_index = pd.date_range(
        start='2020-01-01 00:00',  periods=200, freq='h')

    # Generators
    input_generators = pd.DataFrame(
        columns=['area', 'p_min', 'p_max', 'min_up_time',
                 'min_dn_time', 'startup_costs', 'p_must_run'],
        data=[['de', 100, 500, 200, 100, 0, 0],
              ['de', 100, 500, 150, 100, 0, 0]],
        index=['coal', 'gas'])
    input_generators.index.name = 'name'
    # Load
    input_load = pd.DataFrame(
        data={'de': [1000] * 100 + [500] * 100},
        index=time_index)
    input_load.index.name = 'time'
    # Variable costs
    input_var_costs = pd.DataFrame(
        data={'de_coal': [15] * len(time_index),
              'de_gas': [30] * len(time_index)},
        index=time_index)
    input_var_costs.index.name = 'time'

    input_data = {
        'generators': input_generators,
        'load': input_load,
        'var_costs': input_var_costs}

    return input_data


@pytest.fixture
def dispatch_input_one_area_dn_time():
    '''One area data-set with longer time series and long minimum down
    times in order to further test the rolling horizon optimization
    and the modeling of minimum down time.'''

    time_index = pd.date_range(
        start='2020-01-01 00:00',  periods=200, freq='h')

    # Generators
    input_generators = pd.DataFrame(
        columns=['area', 'p_min', 'p_max', 'min_up_time',
                 'min_dn_time', 'startup_costs', 'p_must_run'],
        data=[['de', 0, 500, 1, 1, 0, 0],
              ['de', 100, 500, 1, 100, 0, 0],
              ['de', 100, 500, 1, 1, 0, 0]],
        index=['coal', 'gas_long_dn_time', 'gas'])
    input_generators.index.name = 'name'
    # Load
    input_load = pd.DataFrame(
        data={'de': [1500] * 10 + [50] * 10 + [1000] * 180},
        index=time_index)
    input_load.index.name = 'time'
    # Variable costs
    input_var_costs = pd.DataFrame(
        data={'de_coal': [15] * len(time_index),
              'de_gas_long_dn_time': [30] * len(time_index),
              'de_gas': [40 * len(time_index)]},
        index=time_index)
    input_var_costs.index.name = 'time'

    input_data = {
        'generators': input_generators,
        'load': input_load,
        'var_costs': input_var_costs}

    return input_data


@pytest.fixture
def dispatch_input_three_areas():
    '''Three area data-set without any storages. Mainly used for testing
    cross-border flows. The following situations should occur:
        t01: fr (only producer) exports to de and at (unlimited NTCs)
        t02: fr-de becomes restricted and de must produce as well
    '''
    # Generators
    input_generators = pd.DataFrame(
        columns=['area', 'p_min', 'p_max', 'min_up_time',
                 'min_dn_time', 'startup_costs', 'p_must_run'],
        data=[['de', 300, 800, 2, 2, 0, 0],
              ['fr', 400, 1000, 10, 10, 0, 0]],
        index=['coal', 'nuc'])
    input_generators.index.name = 'name'
    # Load
    input_load = pd.DataFrame(
        data={'de': [400, 400],
              'fr': [400, 400],
              'at': [100, 100]},
        index=['t01', 't02'])
    input_load.index.name = 'time'
    # Variable costs
    input_var_costs = pd.DataFrame(
        data={'de_coal': [15] * 2, 'fr_nuc': [5] * 2},
        index=['t01', 't02'])
    input_var_costs.index.name = 'time'
    # NTCs
    input_ntc = pd.DataFrame(
        columns=['area1', 'area2', 'ntc'],
        data=[['de', 'fr', 1000],
              ['de', 'at', 1000],
              ['de', 'fr', 100],
              ['de', 'at', 1000]],
        index=['t01', 't01', 't02', 't02'])
    input_ntc.index.name = 'time'

    input_data = {
        'generators': input_generators,
        'load': input_load,
        'var_costs': input_var_costs,
        'ntcs': input_ntc}

    return input_data


@pytest.fixture
def dispatch_input_only_storage():
    '''One area data-set for testing storages.'''

    # Storages
    input_storages = pd.DataFrame(
        columns=['area', 'cap_ini', 'cap_max', 'efficiency', 'power'],
        data=[['de', 100, 300, 0.75, 100]],
        index=['pump1'])
    input_storages.index.name = 'name'
    # Load
    input_load = pd.DataFrame(
        data={'de': [10, -20, 50, 50]},
        index=['t01', 't02', 't03', 't04'])
    input_load.index.name = 'time'

    input_data = {'storages': input_storages, 'load': input_load}

    return input_data


@pytest.fixture
def dispatch_input_only_storage_long():
    '''One area data-set for testing storages with rolling horizon
    optimization.'''

    time_index = pd.date_range(
        start='2020-01-01 00:00',  periods=500, freq='h')

    # Storages
    input_storages = pd.DataFrame(
        columns=['area', 'cap_ini', 'cap_max', 'efficiency', 'power'],
        data=[['de', 1250, 1500, 0.5, 100]],
        index=['pump1'])
    input_storages.index.name = 'name'
    # Load
    input_load = pd.DataFrame(
        data={'de': [-10, 10] * 250},
        index=time_index)
    input_load.index.name = 'time'

    input_data = {'storages': input_storages, 'load': input_load}

    return input_data

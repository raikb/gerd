'''Contains models (so far only Dispatch) defined as classes.

Dispatch is a multi-area power market model that is capable of modeling:
    Power prices
    Power plant dispatch (minimum up-time and down-time of generators)
    Storages (batteries, pump storages etc.)
    Must-run production
    Cross-border flows

The model is solved applying mixed-integer programming (MIP),
relaxed MIP (RMIP) or both consecutively. If the focus is on dispatch,
take MIP. If the focus is on prices and not so much on the binary
dispatch variables, take RMIP. If both are important, let the model run both.
If computation time is the dominating factor, use only RMIP.'''
import logging
import math
import numpy as np
import pandas as pd
# gerd modules
from gerd import input_check
from gerd import model_solver

# Logging
logger = logging.getLogger(__name__)


# %% Constants

_MIN_NR_OPT_HORIZONS = 120
_MIN_NR_CUT_OFF_STEPS = 24


# %% Helpers

def _check_if_column_exists(df: pd.DataFrame, column_name: str):
    '''Returns true if column name is in the DataFrame, otherwise false.
    A warning is returned to the logger if false.

    :param df: pandas DataFrame
    :param column_name: Column name'''

    if column_name in df.columns:
        return True
    else:
        logger.warning(f'Column {column_name} not found in input data.'
                       'Constraint will not be set.')
        return False


def _get_on_ini(
        on_all: pd.Series, idx_ts_new: pd.date_range,
        min_up_time: int, min_dn_time: int):
    '''A rolling horizon optimization requires generator stats
    to be handed over to the next horizon as an initial state. This
    function computes this for the on state based on previous states (on_all),
    the timestamps of the next horizon and the minimum up/down time.
    This function takes one generator (pandas Series) and returns a pandas
    Series,

    :param on_all: Previous on states for one generator as pd.Series
    :param idx_ts_new: Time index of the next horizon
    :param min_up_time: Minimum up time
    :param min_dn_time: Minimum down time
    :rtype: pd.Series'''

    # Cut-off on states
    on_all = on_all[:idx_ts_new[0]].copy()
    on_trans = on_all[-1:]  # Save the transition date for later
    on_all = on_all[:-1]  # Take away the transition date

    # Count consecutively appearing values
    # I.e.: # 0, 0, 1, 1, 1, 0 -> 1, 2, 1, 2, 3, 1
    consecutive_value_count = (
        on_all.groupby((on_all != on_all.shift()).cumsum()).cumcount() + 1)

    # For how many timestamps is the generator on or off
    # 4 means it has been on for 4 timestamps
    on_for = on_all * consecutive_value_count
    off_for = (1-on_all) * consecutive_value_count

    # Get forced initial on/off states
    on_ini = pd.Series(dtype=int)
    data = list()

    # On states to append
    if on_for[-1] > 0:
        data = data + [1] * int(min_up_time - on_for[-1])
    # Off states to append
    elif off_for[-1] > 0:
        data = data + [0] * int(min_dn_time - off_for[-1])

    # Any data?
    if len(data) > 0:
        series_len = min(len(data), len(idx_ts_new))
        on_ini = on_ini.append(
            pd.Series(
                data=data[:series_len],
                index=idx_ts_new[:series_len], dtype=int))
    # If not, take what is already know about the first timestamp
    else:
        on_ini = on_ini.append(on_trans)

    on_ini.name = on_all.name

    return on_ini


# %% Dispatch

class Dispatch:
    '''Multi-area dispatch model. input_data is a dict that holds the
    input data and is the only mandatory input argument. The model will
    be build and solved depending on what is provided in input_data.
    All other input parameters, e.g. model_min_up_time, can be used
    to switch on or off constraints.

    :param input_data: dictionary with input data. Permitted keys are
                       generators, storages, load, var_costs and ntcs.
    :param model_generators: Modeling generators
                             True (default if data provided)/False
    :param model_storages: Modeling storages
                           True (default if data provided)/False
    :param model_min_up_time: Modeling minimum up-time of generators
                              True (default if data provided)/False
    :param model_min_dn_time: Modeling minimum down-time of generators
                              True (default if data provided)/False
    :param model_exchange: Modeling cross-border flows
                           True (default if data provided)/False
    :param model_must_run: Modeling must-run production
                           True (default if data provided)/False'''

    def __init__(self, input_data: dict,
                 model_generators: bool = True,
                 model_storages: bool = True,
                 model_min_up_time: bool = True,
                 model_min_dn_time: bool = True,
                 model_exchange: bool = True,
                 model_must_run: bool = True):
        '''Constructor method'''

        # Process input data - generators (optional)
        if ('generators' in input_data) and model_generators:
            self.input_gens = input_check.check_input_df(
                input_data, 'generators')
            self.model_generators = True
            self.idx_gens = self.input_gens.index

            # Variable costs
            self.input_var_costs = input_check.check_input_df(
                input_data, 'var_costs')

            # Minimum up-time
            if model_min_up_time:
                self.model_min_up_time = _check_if_column_exists(
                    self.input_gens, 'min_up_time')
            else:
                self.model_min_up_time = model_min_up_time

            # Minimum down-time
            if model_min_dn_time:
                self.model_min_dn_time = _check_if_column_exists(
                    self.input_gens, 'min_dn_time')
            else:
                self.model_min_dn_time = model_min_dn_time

            # Must-run option
            if model_must_run:
                self.model_must_run = _check_if_column_exists(
                    self.input_gens, 'p_must_run')

            logger.info('Generator input data provided and checked')

        elif ('generators' not in input_data) and model_generators:
            logger.warning('No generator input data provided')
            self.model_generators = False
        else:
            logger.debug('No generator modeling intended')
            self.model_generators = False

        # Process input data - storages (optional)
        if ('storages' in input_data) and (model_storages is True):
            self.input_stos = input_check.check_input_df(
                input_data, 'storages')
            self.model_storages = True
            self.idx_stos = self.input_stos.index
            logger.info('Storage input data provided and checked')
        elif ('storages' not in input_data) and model_storages:
            logger.warning('No storage input data provided')
            self.model_storages = False
        else:
            logger.debug('No storage modeling intended')
            self.model_storages = False

        # Either storages or generators should be defined
        if (not self.model_generators) and (not self.model_storages):
            raise ValueError('Either storages or generators have to be '
                             'defined. Check your input!')

        # Process input data - NTCs (optional)
        if ('ntcs' in input_data) and (model_exchange is True):
            self.input_ntcs = input_check.check_input_df(
                input_data, 'ntcs')
            self.model_exchange = True
        elif ('ntcs' in input_data) and (model_exchange is False):
            raise ValueError('NTC input data defined but model_exchange '
                             'is set to false. Please remove NTC input '
                             'data or set model_exchange to true.')
        else:
            logger.info('No NTC input data provided')
            self.model_exchange = False

        # Process input data - load (mandatory)
        self.input_load = input_check.check_input_df(
                input_data, 'load')

        # Extract time index (self.idx_ts) from the data
        self._get_unique_time_index_from_input_data()

        # Extract areas from the data
        self._get_areas_from_input_data()

        # Extract topology from the data
        if self.model_exchange:
            self.topology = list(set(zip(self.input_ntcs['area1'],
                                         self.input_ntcs['area2'])))

    def _get_areas_from_input_data(self):
        '''Looks for all defined areas in input_data and gets them
        as a unique tuple.
        Area information is contained in the generators, storage, load
        and exchange (NTC) input.'''

        self.areas = list()
        # Append areas in generator input
        if self.model_generators:
            self.areas.append(self.input_gens['area'].tolist())
        # Append areas in storage input
        if self.model_storages:
            self.areas.append(self.input_stos['area'].tolist())
        # Append areas in NTC input
        if self.model_exchange:
            self.areas.append(self.input_ntcs['area1'].tolist())
            self.areas.append(self.input_ntcs['area2'].tolist())
        # Append areas in load input
        self.areas.append(self.input_load.columns.tolist())
        # Flatten list and take only unique values
        self.areas = set([item for sublist in self.areas for item in sublist])
        self.areas = tuple(self.areas)

        logger.info(f'Areas found in the input: {[a for a in self.areas]}')

    def _get_unique_time_index_from_input_data(self):
        '''Looks for all defined timestamps in input_data and gets them
        as a unique index.
        Time index is contained in the load, variable costs
        and exchange (NTC) input.'''

        self.idx_ts = self.input_load.index.unique()
        if self.model_generators:
            self.idx_ts = self.idx_ts.intersection(
                self.input_var_costs.index).unique()
        if self.model_exchange:
            self.idx_ts = self.idx_ts.intersection(
                self.input_ntcs.index).unique()

        logger.info(f'{self.idx_ts.size} common timestamps found in the input')

    def return_solution_from_solver(self, solver):
        '''The class ModelSolver returns results as pandas DataFrames.
        This function gets and stores them in a dict.'''

        solution = dict()

        if self.model_generators:
            solution.update({'prod': solver.get_solution_values('prod')})
            if self.model_type == 'mip_rmip':
                solution.update(
                    {'on': solver.get_solution_values('on_mip')})
                solution.update(
                    {'start': solver.get_solution_values('start_mip')})
                solution.update(
                    {'stop': solver.get_solution_values('stop_mip')})
            else:
                solution.update({'on': solver.get_solution_values('on')})
                solution.update({'start': solver.get_solution_values('start')})
                solution.update({'stop': solver.get_solution_values('stop')})

        if self.model_storages:
            solution.update(
                {'charge': solver.get_solution_values('charge')})
            solution.update(
                {'discharge': solver.get_solution_values('discharge')})
            solution.update(
                {'storage_cap': (self.input_stos['cap_ini']
                                 + (solution['charge'].cumsum()
                                 * self.input_stos['efficiency'])
                                 - solution['discharge'].cumsum())})

        if self.model_exchange:
            solution.update({'exc': solver.get_solution_values('exc')})

        if self.model_type != 'mip':
            solution.update({'prices': solver.get_solution_values('prices')})

        return solution

    def optimize(
            self, model_type: str = 'mip_rmip',
            rolling_optimization: bool = False,
            **rolling_opt_parameters):
        '''Hands the model over to ModelSolver, solves it and returns
        the results as pandas DataFrames as well as the objective value.
        The mip option does not return dual values and thus no prices.
        If the focus is on dispatch, take mip. If the focus is also on
        prices take mip_rmip.

        :param model_type: How to solve the model.
                           Options are: mip, mip_rmip (default)
        :param rolling_optimization: Apply rolling horizon optimization
                                     (True/False (default))'''

        # Solver type
        self.model_type = input_check.check_model_type(
            model_type, self.model_generators)

        # Optimize
        if rolling_optimization:
            if len(self.idx_ts) <= _MIN_NR_OPT_HORIZONS:
                logger.warning('Data set too small for a rolling '
                               'horizon optimization. Single optimization '
                               'is taken instead.')
                self.solution = self._optimize_single()
            else:
                # Check timestamp index
                if not isinstance(self.idx_ts, pd.DatetimeIndex):
                    raise TypeError('Other index data types than pandas '
                                    'DatetimeIndex are not yet supported '
                                    'for rolling optimization.')
                else:
                    logger.warning('Only hourly input data supported '
                                   'at the moment')
                    self.solution = self._optimize_rolling(
                        **rolling_opt_parameters)
        else:
            self.solution = self._optimize_single()

    def _optimize_single(self):
        '''Optimizes the model in one step, saves the objective value
        and returns the solution.'''

        solver = model_solver.ModelSolver(self)

        self.objective_value = solver.get_objective_value()

        return self.return_solution_from_solver(solver)

    def _optimize_rolling(self, **rolling_opt_parameters):
        '''Optimized the problem applying rolling optimization. The number
        of horizons and the number of cut-off time steps can be fined.
        Otherwise, they are estimated.'''

        # Take optional optimization parameters or estimate them
        if 'nr_opt_horizons' in rolling_opt_parameters.keys():
            nr_opt_horizons = rolling_opt_parameters['nr_opt_horizons']
        else:
            nr_opt_horizons = self._compute_nr_of_rolling_horizons()
            logger.debug('Number of optimization horizons was computed '
                         f'as {nr_opt_horizons}')
        self.nr_opt_horizons = nr_opt_horizons

        if 'nr_cut_off_steps' in rolling_opt_parameters.keys():
            nr_cut_off_steps = rolling_opt_parameters['nr_cut_off_steps']
        else:
            nr_cut_off_steps = self._compute_nr_of_cut_off_steps(
                nr_opt_horizons)
            logger.debug('Number of cut-off time steps was computed '
                         f'as {nr_cut_off_steps}')
        self.nr_cut_off_steps = nr_cut_off_steps

        # Split the data set in the rolling horizons
        idx_ts_split = np.array_split(self.idx_ts, self.nr_opt_horizons)

        # Run first horizon
        model_h = self
        model_h.idx_ts = idx_ts_split[0]
        solver = model_solver.ModelSolver(model_h)

        # Extract solutions for this horizon and write them in a dict
        solution_h = self.return_solution_from_solver(solver)

        # Cut-off solutions (last nr_cut_off_steps steps not needed)
        for sol_key, sol_df in solution_h.items():
            solution_h[sol_key] = sol_df[:-self.nr_cut_off_steps]

        # Append solution of this horizon to the final solution to return it
        solution_final = solution_h.copy()

        # Go through all remaining horizons
        for h, h_idx_ts in enumerate(idx_ts_split[1:], 1):
            logger.info(f'Started horizon {h + 1} from {self.nr_opt_horizons}')
            # Hand over the model for this horizon
            model_h = self

            # The time index of the last horizon
            h_idx_ts_last = idx_ts_split[h-1]
            # Take only the cut-off period because it will be the beginning
            # of this horizon and hence computed twice
            h_idx_ts_last_cutoff = h_idx_ts_last[-(
                self.nr_cut_off_steps + 1):]

            # Append current horizon to the time index
            model_h.idx_ts = h_idx_ts_last_cutoff.append(h_idx_ts)

            # Get previous system states to hand over
            if self.model_generators:
                # Get on_ini (on/off states for the current period) based
                # on all foregone on/off states.
                on_all = solution_final['on'].copy()
                on_ini = on_all.apply(
                    lambda x: _get_on_ini(
                        x,
                        model_h.idx_ts,
                        self.input_gens.loc[x.name, 'min_up_time'],
                        self.input_gens.loc[x.name, 'min_dn_time']))

            if self.model_storages:
                # Initial storage capacities for the next horizon. Must be
                # taken from the timestamp before the transition timestamp.
                # That is why iloc[-2] is taken.
                model_h.input_stos['cap_ini'] = solution_final[
                    'storage_cap'].iloc[-2].copy()

            # Solve
            if self.model_generators:
                solver = model_solver.ModelSolver(
                    model_h, on_ini=on_ini)
            else:
                solver = model_solver.ModelSolver(model_h)

            # Extract solutions for this horizon and write them in a dict
            solution_h = self.return_solution_from_solver(solver)

            # Cut-off solutions (last nr_cut_off_steps steps not needed)
            if (h + 1) == self.nr_opt_horizons:
                # The overlapping timestamp is computed twice. Once
                # at the end of the foregone horizon and again at the
                # beginning of the current horizon. That is why,
                # the first value is not taken here.
                for sol_key, sol_df in solution_h.items():
                    solution_h[sol_key] = sol_df[1:]
            else:
                # Same here with the first value
                # (cf. h == self.nr_opt_horizons).
                for sol_key, sol_df in solution_h.items():
                    solution_h[sol_key] = sol_df[1:-self.nr_cut_off_steps]

            # Append solution of this horizon to the final solution
            for sol_key, sol_df in solution_h.items():
                solution_final[sol_key] = solution_final[sol_key].append(
                    solution_h[sol_key])

        return solution_final

    def _compute_nr_of_rolling_horizons(self):
        '''Computes the number of rolling optimization horizons if they
        are not defined by the user. The minimum is 120 time steps or
        2 horizons.
        Generators and storage parameters are taken into account as well.
        Regarding generators, one horizon should be at least as long as
        the maximum minimum up or down time of all generators.
        Regarding storages, one horizon should be at least as long as it takes
        to empty a storage. This can be computed by dividing the storage
        capcity with the power.'''

        # Start with the minimum
        len_opt_horizons = [_MIN_NR_OPT_HORIZONS]

        # Append maximum values based on the generator input
        if self.model_generators:
            if self.model_min_dn_time and self.model_min_up_time:
                len_opt_horizons.append(
                    self.input_gens[['min_dn_time',
                                    'min_up_time']].max().max())
            elif self.model_min_dn_time:
                len_opt_horizons.append(self.input_gens['min_dn_time'].max())
            elif self.model_min_up_time:
                len_opt_horizons.append(self.input_gens['min_up_time'].max())

        # Append maximum values based on the storage input
        if self.model_storages:
            len_opt_horizons.append(
                (self.input_stos['cap_max']
                 / self.input_stos['power']).max() * 2)

        nr_opt_horizons = max(
            2, math.floor(len(self.idx_ts) / max(len_opt_horizons)))

        return nr_opt_horizons

    def _compute_nr_of_cut_off_steps(self, nr_opt_horizons):
        '''Computes the number of cut-off timestamps based on
        the number of rolling horizons.'''

        nr_cut_off_steps = max(
            _MIN_NR_CUT_OFF_STEPS,
            math.ceil(nr_opt_horizons * 0.25))

        return nr_cut_off_steps

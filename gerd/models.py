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
import itertools
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


def _get_last_generator_action(
        solution: dict, min_up_time: pd.Series, min_dn_time: pd.Series):
    '''This function returns the timestamps of the last generator
    actions (start and stop). Based on the minimum up-time and down-time,
    it is calculated until which timestamp the generator needs to be
    on or off.'''

    # Initialize the DataFrame that contains all generators' last actions
    last_action = pd.DataFrame(
        index=solution['start'].columns,
        columns=['start', 'stop', 'keep_on_till', 'keep_off_till'],
        data=np.nan)

    # Add minimum up and down times
    last_action['min_up_time'] = min_up_time.copy()
    last_action['min_dn_time'] = min_dn_time.copy()

    # Take the last start and and stop if existing
    for c in solution['start'].columns:
        start_actions = solution['start'].index[solution['start'][c] != 0]
        if not start_actions.empty:
            last_action.loc[c, 'start'] = start_actions[-1]  # Last one

        stop_actions = solution['stop'].index[solution['stop'][c] != 0]
        if not stop_actions.empty:
            last_action.loc[c, 'stop'] = stop_actions[-1]  # Last one

    # Compute until generators need to be on or off
    if not all(last_action['start'].isna()):
        # last_action['keep_on_till'] = (
        #     last_action['start'] + pd.to_timedelta(min_up_time, unit='h'))
        last_action['keep_on_till'] = last_action.apply(
            lambda x: x['start'] + pd.to_timedelta(
                x['min_up_time'] - 1, unit='h'), axis=1)

    if not all(last_action['stop'].isna()):
        # last_action['keep_off_till'] = (
        #     last_action['stop'] + pd.to_timedelta(min_dn_time, unit='h'))
        last_action['keep_off_till'] = last_action.apply(
            lambda x: x['stop'] + pd.to_timedelta(
                x['min_dn_time'] - 1, unit='h'), axis=1)

    return last_action


def _get_initial_on_states(last_action: pd.DataFrame):
    '''A rolling horizon optimization requires generator stats
    to be handed over to the next horizon as an initial state. This
    function computes this for the on state based on last_action
    and returns a pandas DataFrame.

    :param last_action: Output of _get_last_generator_action()
    :rtype: pd.DataFrame'''

    # Get relevant timestamps for on_ini
    t_first = min([pd.to_datetime(last_action['start']).min(),
                  pd.to_datetime(last_action['stop']).min()])
    t_last = max([pd.to_datetime(last_action['keep_on_till']).max(),
                 pd.to_datetime(last_action['keep_off_till']).max()])
    # Initialize on_ini
    on_ini = pd.DataFrame(
        index=pd.date_range(t_first, t_last, freq='h'),
        columns=last_action.index,
        data=np.nan)

    # Fill with on data (0 or 1)
    # Power plants that must be kept on
    if ('keep_on_till' in last_action.columns):
        for index, row in last_action.iterrows():
            if isinstance(row['start'], pd.Timestamp):
                on_ini.loc[row['start']:row['keep_on_till'], index] = 1
    # Power plants that must be kept off
    if 'keep_off_till' in last_action.columns:
        for index, row in last_action.iterrows():
            if isinstance(row['stop'], pd.Timestamp):
                on_ini.loc[row['stop']:row['keep_off_till'], index] = 0

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
            self, model_type: str = 'rmip',
            rolling_optimization: bool = False,
            **rolling_opt_parameters):
        '''Hands the model over to ModelSolver, solves it and returns
        the results as pandas DataFrames as well as the objective value.
        The mip option does not return dual values and thus no prices.
        If the focus is on dispatch, take mip. If the focus is on prices
        and not so much on the binary dispatch variables, take rmip.
        If both are important, take mip_rmip.

        :param model_type: How to solve the model.
                           Options are: mip, rmip (default), mip_rmip
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
                                   ' at the moment')
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

        # Save a copy of cap_ini of storages because it will be changed
        if self.model_storages:
            cap_ini_copy = self.input_stos['cap_ini'].copy()

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

        # Go through all horizons
        for h, h_idx_ts in enumerate(idx_ts_split, 1):
            logger.info(f'Started horizon {h} from {self.nr_opt_horizons}')
            # Hand over the model for this horizon
            model_h = self

            # Solve normally if it is the first timestamp
            if h == 1:
                model_h.idx_ts = h_idx_ts
                solver = model_solver.ModelSolver(model_h)
            # If after the first horizon, hand over previous
            # variable states to the solver
            else:
                # Get the cut-off from the previous horizon
                # because we have to optimize these timestamps again
                h_before_cutoff = h_idx_ts_before[-(
                    self.nr_cut_off_steps + 1):]
                # Append current horizon to the time index
                model_h.idx_ts = h_before_cutoff.append(h_idx_ts)
                # Get previous on states and hand over
                # Relevant on states can be irrelevant if decisions have
                # been made a long time ago and up and down times are no
                # longer relevant
                if self.model_generators:
                    if not on_ini.index.intersection(model_h.idx_ts).empty:
                        # Take only timestamps modeled in h
                        on_ini = on_ini.loc[
                            on_ini.index.intersection(model_h.idx_ts)]
                        # Add on_ini state of the first hour
                        on_ini.iloc[0] = solution_final[
                            'on'].iloc[-1]
                        # Solve
                        solver = model_solver.ModelSolver(
                            model_h, prod_ini=prod_ini, on_ini=on_ini)
                    else:
                        # Solve
                        solver = model_solver.ModelSolver(
                            model_h,
                            prod_ini=prod_ini,
                            on_ini=solution_final['on'].iloc[-1].to_frame().T)
                else:
                    solver = model_solver.ModelSolver(model_h)

            # Extract solutions for this horizon and write them in a dict
            solution_h = self.return_solution_from_solver(solver)

            # Cut-off solutions (last nr_cut_off_steps steps not needed)
            if h == self.nr_opt_horizons:
                # The overlapping timestamp is computed twice. Once
                # at the end of the foregone horizon and again at the
                # beginning of the current horizon. That is why,
                # the first value is not taken here.
                for sol_key, sol_df in solution_h.items():
                    solution_h[sol_key] = sol_df[1:]
            elif h == 1:
                for sol_key, sol_df in solution_h.items():
                    solution_h[sol_key] = sol_df[:-self.nr_cut_off_steps]
            else:
                # Same here with the first value
                # (cf. h == self.nr_opt_horizons).
                for sol_key, sol_df in solution_h.items():
                    solution_h[sol_key] = sol_df[1:-self.nr_cut_off_steps]

            # Append solution of this horizon to the final solution
            if h == 1:
                solution_final = solution_h.copy()
            else:
                for sol_key, sol_df in solution_h.items():
                    solution_final[sol_key] = solution_final[sol_key].append(
                        solution_h[sol_key])

            # Initial generator states for next horizon (on_ini, prod_ini)
            if self.model_generators:
                last_action = _get_last_generator_action(
                    solution_final, self.input_gens['min_up_time'],
                    self.input_gens['min_dn_time'])
                on_ini = _get_initial_on_states(last_action)

                prod_ini = solution_final['prod'].iloc[-1]

            # Initial storage capacities for the next horizon.
            # Needs to be changed in self because input_stos is taken
            # from self to compute the storage capacity in
            # return_solution_from_solver()
            if self.model_storages:
                self.input_stos['cap_ini'] = solution_final[
                    'storage_cap'].iloc[-1].copy()

            # Remember last horizon's time index for cutting off
            h_idx_ts_before = h_idx_ts

        # Return initial cap_ini
        if self.model_storages:
            self.input_stos['cap_ini'] = cap_ini_copy

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

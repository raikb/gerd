'''Contains the actual formulation of problem. The class ModelSolver builds
and solves a model. ModelSolver contains functions for defining variables,
constraints and the objective function.
All variables are in 2-dimensional nested dicts, in which the first level
is always the timestamp index. Data that is input to ModelSolver is usually
a pandas DataFrame, in which the index contains the timestamp.'''

import itertools
import logging
import numpy as np
from ortools.linear_solver import pywraplp
import pandas as pd
import sys

# Logging
logger = logging.getLogger(__name__)


# %% Helpers
def get_pywraplp_solver_status_msg(status):

    if status == pywraplp.Solver.OPTIMAL:
        return 'optimal'
    elif status == pywraplp.Solver.INFEASIBLE:
        return 'proven infeasible'
    elif status == pywraplp.Solver.FEASIBLE:
        return 'feasible, or stopped by limit'
    elif status == pywraplp.Solver.UNBOUNDED:
        return 'proven unbounded'
    elif status == pywraplp.Solver.ABNORMAL:
        return 'abnormal, i.e., error of some kind'
    else:
        raise ValueError('Solver status cannot be decoded.')


def _init_2d_nested_dict(keys1, keys2):
    '''Creates 2-dimensional nested dict that are meant to hold a variable.

    :param keys1: Keys for the first level as a list-like element
    :param keys2: Keys for the second level as a list-like element
    '''
    # First level
    var = dict.fromkeys(keys1)
    # Second level
    for k1 in keys1:
        var[k1] = dict.fromkeys(keys2)
    return var


# %% ModelSolver

class ModelSolver:
    '''Builds and solves a model. ModelSolver contains functions for
    defining variables, constraints and the objective function. The input
    is an object from a class in gerd.models and optionally initial
    parameters.'''

    def __init__(self, model, **initial_values):
        '''Initializes ModelSolver and optimized directly.'''

        self.idx_ts_type = type(model.idx_ts[0])

        # Initial values (only used with rolling optimization)
        self.initial_values = initial_values

        # if 'prod_ini' in self.initial_values.keys():
        #     self.has_ini_prod_values = True
        # else:
        #     self.has_ini_prod_values = False

        if 'on_ini' in self.initial_values.keys():
            self.has_ini_on_values = True
        else:
            self.has_ini_on_values = False

        # Dispatch model
        if model.__class__.__name__ == 'Dispatch':

            if model.model_type == 'mip_rmip':
                # Solve MIP and RMIP
                self.build_and_optimize_mip_rmip(model)

            elif model.model_type == 'mip':
                # Solve MIP
                self.build_and_optimize_mip(model)

        else:
            raise ValueError(f'Unknown model type {model.model_type}.')

    def set_solver(self, problem_type: str):
        '''Sets the solver based on the problem type (rmip or mip).

        :param problem_type: Problem type (rmip or mip)'''

        if problem_type == 'rmip':
            self.solver = pywraplp.Solver(
                'gerd_linear_dispatch',
                pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        elif problem_type == 'mip':
            self.solver = pywraplp.Solver(
                'gerd_linear_dispatch',
                pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        else:
            raise ValueError('Unknown solver type. Use "linear" or "mip".')

        self.status = None  # No solver status yet

        logger.debug('Initialized solver')

    def build_dispatch_problem(self, model):
        '''Initializes and defines variables, adds constraints depending
        on the input and defines the objective function for a
        dispatch problem. The solver is expected to be set before.'''

        # Initialize variables
        if model.model_generators:
            self.init_gen_variables(model)
        if model.model_storages:
            self.init_sto_variables(model)
        if model.model_exchange:
            self.init_exchange_variables(model)

        # Add generator constraints (optional)
        if model.model_generators:
            self.add_constr_prod_limits(model)
            self.add_constr_gen_logic(model)
            if model.model_min_up_time:
                self.add_constr_min_up_time(model)
            if model.model_min_dn_time:
                self.add_constr_min_dn_time(model)
            if model.model_must_run:
                self.add_constr_must_run(model)

        # Add storage constraints (optional)
        if model.model_storages:
            self.add_constr_storages(model)

        # Add exchange constraints (optional)
        if model.model_exchange:
            self.add_constr_exchange(model)

        # Add common constraints and the objective function
        self.add_constr_energy_balance(model)
        self.add_objective_function(model)

    def init_gen_variables(self, model):
        '''Initializes nested dicts that hold all generator variables
        and adds them to the solver including their bounds.'''

        # Create nested dicts
        self.var_prod = _init_2d_nested_dict(model.idx_ts, model.idx_gens)
        self.var_on = _init_2d_nested_dict(model.idx_ts, model.idx_gens)
        self.var_stop = _init_2d_nested_dict(model.idx_ts, model.idx_gens)
        self.var_start = _init_2d_nested_dict(model.idx_ts, model.idx_gens)

        # Add variables to the solver
        for t, g in itertools.product(model.idx_ts, model.idx_gens):
            self.var_prod[t][g] = self.solver.NumVar(
                lb=0, ub=float(model.input_gens.loc[g, 'p_max']),
                name=f'gen_{t}_{g}')
            self.var_on[t][g] = self.solver.IntVar(
                lb=0, ub=1, name=f'on_{t}_{g}')
            self.var_stop[t][g] = self.solver.IntVar(
                lb=0, ub=1, name=f'stop_{t}_{g}')
            self.var_start[t][g] = self.solver.IntVar(
                lb=0, ub=1, name=f'start_{t}_{g}')

        logger.debug('Initialized generator variables: var_prod, '
                     'var_on, var_stop, var_start')

    def init_sto_variables(self, model):
        '''Initializes nested dicts that hold all storage variables
        and adds them to the solver including their bounds.'''

        # Create nested dicts
        self.var_charge = _init_2d_nested_dict(model.idx_ts, model.idx_stos)
        self.var_discharge = _init_2d_nested_dict(model.idx_ts, model.idx_stos)

        # Add variables to the solver
        for t, s in itertools.product(model.idx_ts, model.input_stos.index):
            self.var_charge[t][s] = self.solver.NumVar(
                    lb=0, ub=float(model.input_stos.loc[s, 'power']),
                    name=f'charge_{t}_{s}')
            self.var_discharge[t][s] = self.solver.NumVar(
                    lb=0, ub=float(model.input_stos.loc[s, 'power']),
                    name=f'discharge_{t}_{s}')

        logger.debug('Initialized storage variables: var_charge, '
                     'var_discharge')

    def init_exchange_variables(self, model):
        '''Initializes a nested dicts that holds the exchange
        variable and adds it to the solver including its bounds.'''

        # Create nested dict
        self.var_exc = _init_2d_nested_dict(model.idx_ts, model.topology)

        # Add variables to the solver
        for t, topo in itertools.product(model.idx_ts, model.topology):
            is_area_from = model.input_ntcs['area1'] == topo[0]
            is_area_to = model.input_ntcs['area2'] == topo[1]
            is_time = model.input_ntcs.index == t
            is_all = is_area_from & is_area_to & is_time
            ntc_value = float(model.input_ntcs.loc[is_all, 'ntc'])
            self.var_exc[t][topo] = self.solver.NumVar(
                    lb=-ntc_value, ub=ntc_value, name=f'exc_{t}_{topo}')

        logger.debug('Initialized exchange variable: var_exc')

    def add_constr_prod_limits(self, model):
        '''Adds constraint: Generator production limits.'''

        for t in model.idx_ts:
            for row in model.input_gens.itertuples():
                self.solver.Add(self.var_prod[t][row.Index] <=
                                row.p_max * self.var_on[t][row.Index])
                self.solver.Add(self.var_prod[t][row.Index] >=
                                row.p_min * self.var_on[t][row.Index])

    def add_constr_exchange(self, model):
        '''Adds constraint: Cross-border flows.'''

        for t, topo in itertools.product(model.idx_ts, model.topology):
            is_area_from = model.input_ntcs['area1'] == topo[0]
            is_area_to = model.input_ntcs['area2'] == topo[1]
            is_time = model.input_ntcs.index == t
            is_all = is_area_from & is_area_to & is_time
            ntc_value = float(model.input_ntcs.loc[is_all, 'ntc'])
            self.solver.Add(self.var_exc[t][topo] <= ntc_value,
                            f'ntc_ub_{t}_{topo}')
            self.solver.Add(self.var_exc[t][topo] >= -ntc_value,
                            f'ntc_lb_{t}_{topo}')

        logger.debug('Added exchange constraints')

    def add_constr_energy_balance(self, model):
        '''Adds constraint: Demand equals production in each timestamp.'''

        for t in model.idx_ts:
            for a in model.areas:
                # Production of generators
                if model.model_generators:
                    expr_generation = [self.var_prod[t][g]
                                       for g in model.idx_gens
                                       if g.split('_')[0] == a]
                else:
                    expr_generation = []
                # Storage charging and discharging
                if model.model_storages:
                    expr_sto_charging = [self.var_charge[t][p]
                                         for p in model.idx_stos
                                         if p.split('_')[0] == a]
                    expr_sto_discharging = [self.var_discharge[t][p]
                                            for p in model.idx_stos
                                            if p.split('_')[0] == a]
                else:
                    expr_sto_charging = expr_sto_discharging = []
                # Export and import
                if model.model_exchange:
                    expr_export = [self.var_exc[t][topo]
                                   for topo in model.topology if a == topo[0]]
                    expr_import = [self.var_exc[t][topo]
                                   for topo in model.topology if a == topo[1]]
                else:
                    expr_export = expr_import = []
                # Add constraint
                self.solver.Add(sum(expr_generation)
                                - sum(expr_export)
                                + sum(expr_import)
                                - sum(expr_sto_charging)
                                + sum(expr_sto_discharging)
                                - model.input_load.loc[t, a] == 0,
                                f'balance_{t}_{a}')

    def add_constr_gen_logic(self, model):
        '''Adds constraint: on/off - logic constraint.'''

        for g in model.idx_gens:
            # t=0
            if self.has_ini_on_values:
                if np.isnan(
                        self.initial_values['on_ini'].loc[model.idx_ts[0], g]):
                    self.solver.Add(
                        (self.var_start[model.idx_ts[0]][g] -
                         self.var_stop[model.idx_ts[0]][g]) ==
                        self.var_on[model.idx_ts[0]][g])
            else:
                self.solver.Add((self.var_start[model.idx_ts[0]][g] -
                                self.var_stop[model.idx_ts[0]][g]) ==
                                self.var_on[model.idx_ts[0]][g])
            # t>0
            for t, t_before in zip(model.idx_ts[1:], model.idx_ts[:-1]):
                self.solver.Add((self.var_start[t][g] -
                                self.var_stop[t][g]) ==
                                (self.var_on[t][g] -
                                 self.var_on[t_before][g]))

        logger.debug('Added constraints for the generator logic')

    def add_constr_min_up_time(self, model):
        '''Adds constraint: Minimum up-time time.'''

        for row in model.input_gens.itertuples():
            if model.input_gens.loc[row.Index, 'min_up_time'] != 0:
                for ti, t in enumerate(model.idx_ts):
                    ti_start = max(
                        [0,
                         ti-model.input_gens.loc[row.Index, 'min_up_time']+1])
                    ti_end = (ti+1)
                    expr = [self.var_start[t][row.Index]
                            for t in model.idx_ts[ti_start:ti_end]]
                    self.solver.Add(sum(expr) <= self.var_on[t][row.Index])

        logger.debug('Added constraints for minimum up-time')

    def add_constr_min_dn_time(self, model):
        '''Adds constraint: Minimum down-time time.'''

        for row in model.input_gens.itertuples():
            if model.input_gens.loc[row.Index, 'min_dn_time'] != 0:
                for ti, t in enumerate(model.idx_ts):
                    ti_start = max(
                        [0,
                         ti-model.input_gens.loc[row.Index, 'min_dn_time']+1])
                    ti_end = (ti+1)
                    expr = [self.var_stop[t][row.Index]
                            for t in model.idx_ts[ti_start:ti_end]]
                    self.solver.Add(sum(expr) <= 1 - self.var_on[t][row.Index])

        logger.debug('Added constraints for minimum down-time')

    def add_constr_must_run(self, model):
        '''Adds constraint: Must-run.'''

        for t in model.idx_ts:
            for row in model.input_gens.itertuples():
                if row.p_must_run > 0:
                    self.solver.Add(
                        self.var_prod[t][row.Index] >=
                        row.p_must_run * self.var_on[t][row.Index])

        logger.debug('Added must-run constraints')

    def add_constr_storages(self, model):
        '''Adds constraint: Storages.'''

        # Stored energy <= storage maximum capacity
        for row in model.input_stos.itertuples():
            for ti, _ in enumerate(model.idx_ts, 1):
                expr = [self.var_charge[t][row.Index] * row.efficiency -
                        self.var_discharge[t][row.Index]
                        for t in model.idx_ts[:ti]]
                self.solver.Add(sum(expr) + row.cap_ini <= row.cap_max)
        # discharging - charging <= initial storage capacity
        for row in model.input_stos.itertuples():
            for ti, _ in enumerate(model.idx_ts, 1):
                expr = [self.var_discharge[t][row.Index] -
                        self.var_charge[t][row.Index] * row.efficiency
                        for t in model.idx_ts[:ti]]
                self.solver.Add(sum(expr) <= row.cap_ini)

        logger.debug('Added storage constraints')

    def fasten_variable(self, var: dict, values2fasten: dict):
        '''Fasten a variable by adding a constraint.'''

        for k1 in values2fasten.keys():
            for k2 in values2fasten[k1].keys():
                if not np.isnan(values2fasten[k1][k2]):
                    self.solver.Add(var[k1][k2] == values2fasten[k1][k2])

    def fasten_gen_int_variables(
            self, var_on_res, var_start_res, var_stop_res):
        '''Fastens all integer generator variabels.'''

        self.fasten_variable(self.var_on, var_on_res)
        self.fasten_variable(self.var_start, var_start_res)
        self.fasten_variable(self.var_stop, var_stop_res)

    def fasten_prod_ini(self, prod_ini):
        '''Needs to be DF because of index/orient '''

        values2fasten = prod_ini.to_frame().T.to_dict(orient='index')
        self.fasten_variable(self.var_prod, values2fasten)

    def fasten_on_ini(self, on_ini):
        '''Fastens generation integer variables with a DataFrame as input.'''

        values2fasten = on_ini.to_dict(orient='index')
        self.fasten_variable(self.var_on, values2fasten)

    def add_objective_function(self, model):
        '''Adds the objective function to the solver. The objective
        function includes start-up costs and variable costs if
        generators are modeled.'''
        # Normal definition with generators
        if model.model_generators:
            # Variable costs
            expr_var_costs = [self.var_prod[t][g] *
                              model.input_var_costs.loc[t, g]
                              for t, g in itertools.product(
                                  model.idx_ts, model.idx_gens)]
            # Start-up costs
            expr_startup_costs = [self.var_start[t][g] *
                                  model.input_gens.loc[g, 'startup_costs']
                                  for t, g in itertools.product(
                                      model.idx_ts, model.idx_gens)]
            # Set objective function
            self.solver.Minimize(sum(expr_var_costs) + sum(expr_startup_costs))
        # Only storages
        if not model.model_generators and model.model_storages:
            expr_charging = [self.var_charge[t][g]
                             for t, g in itertools.product(model.idx_ts,
                             model.idx_stos)]
            self.solver.Minimize(sum(expr_charging))

        logger.debug('Set objective function')

    def optimize(self):
        '''Optimizes the problem and adds the objective value
        to self if optimal.'''

        # Solve the system.
        logger.info('Solver started')
        self.status = self.solver.Solve()
        # Report status and time needed
        logger.info('Solver status: '
                    f'{get_pywraplp_solver_status_msg(self.status)}')
        logger.info(f'Time needed: {self.solver.wall_time()/1000} sec.')
        # Exit if solver status not optimal
        if get_pywraplp_solver_status_msg(self.status) != 'optimal':
            sys.exit('Solver status is not optimal. Optimization stopped.')

    def build_and_optimize_mip(self, model):
        '''Initializes and defines variables, adds constraints depending
        on the input, defines the objective function for a
        dispatch MIP problem and solves it.'''

        logger.info('Build and optimize MIP')
        self.set_solver('mip')

        # Initialize variables
        if model.model_generators:
            self.init_gen_variables(model)
        if model.model_storages:
            self.init_sto_variables(model)
        if model.model_exchange:
            self.init_exchange_variables(model)

        # Add generator constraints (optional)
        if model.model_generators:
            self.add_constr_prod_limits(model)
            self.add_constr_gen_logic(model)
            if model.model_min_up_time:
                self.add_constr_min_up_time(model)
            if model.model_min_dn_time:
                self.add_constr_min_dn_time(model)
            if model.model_must_run:
                self.add_constr_must_run(model)

        # Add storage constraints (optional)
        if model.model_storages:
            self.add_constr_storages(model)

        # Add exchange constraints (optional)
        if model.model_exchange:
            self.add_constr_exchange(model)

        # Add common constraints and the objective function
        self.add_constr_energy_balance(model)
        self.add_objective_function(model)

        # Fasten possibly initial variable values
        if self.has_ini_on_values:
            self.fasten_on_ini(self.initial_values['on_ini'])

        self.optimize()

    def build_and_optimize_mip_rmip(self, model):
        '''Initializes and defines variables, adds constraints depending
        on the input, defines the objective function for a
        dispatch MIP problem and solves it. After this, a RMIP is
        build and solved for computing prices.'''

        self.build_and_optimize_mip(model)

        # Save integer variable solutions
        self.var_on_mip = self.get_var_solution_as_dict(self.var_on)
        self.var_start_mip = self.get_var_solution_as_dict(
            self.var_start)
        self.var_stop_mip = self.get_var_solution_as_dict(
            self.var_stop)

        logger.info('Build and optimize RMIP')
        self.set_solver('rmip')

        # Initialize variables
        if model.model_generators:
            self.init_gen_variables(model)
        if model.model_storages:
            self.init_sto_variables(model)
        if model.model_exchange:
            self.init_exchange_variables(model)

        # Add generator constraints (optional)
        if model.model_generators:
            self.add_constr_prod_limits(model)
            if model.model_must_run:
                self.add_constr_must_run(model)

        # Add storage constraints (optional)
        if model.model_storages:
            self.add_constr_storages(model)

        # Add exchange constraints (optional)
        if model.model_exchange:
            self.add_constr_exchange(model)

        # Add common constraints and the objective function
        self.add_constr_energy_balance(model)
        self.add_objective_function(model)

        # Fasten integer variables already optimized in the foregone MIP
        self.fasten_gen_int_variables(
            self.var_on_mip, self.var_start_mip, self.var_stop_mip)

        self.optimize()

    def get_objective_value(self):
        '''Returns the objective value and adds it to self.'''

        if self.status == pywraplp.Solver.OPTIMAL:
            return self.solver.Objective().Value()
        else:
            raise ValueError('Solution not optimal.')

    def get_var_solution_as_dict(self, var):
        '''Works only with 2d nested dicts
        :rtype: dict'''

        var_solution = var.copy()

        for k1 in var.keys():
            for k2 in var[k1].keys():
                var_solution[k1][k2] = var[k1][k2].solution_value()

        return var_solution

    def get_var_solution_as_df(self, var):
        '''Works only with 2d nested dicts
        :rtype: dict'''

        var_solution = self.get_var_solution_as_dict(var)
        df_solution = pd.DataFrame.from_dict(var_solution, orient='index')

        return df_solution

    def get_solution_values(self, var_name: str):
        '''Returns the solution values from a variable as a
        pandas DataFrame.'''

        if var_name == 'prod':
            res = self.get_var_solution_as_df(self.var_prod)
        elif var_name == 'on':
            res = self.get_var_solution_as_df(self.var_on)
        elif var_name == 'on_mip':
            # var_on_mip contains already the solution values
            res = pd.DataFrame.from_dict(self.var_on_mip, orient='index')
        elif var_name == 'start':
            res = self.get_var_solution_as_df(self.var_start)
        elif var_name == 'start_mip':
            # var_start_mip contains already the solution values
            res = pd.DataFrame.from_dict(self.var_start_mip, orient='index')
        elif var_name == 'stop':
            res = self.get_var_solution_as_df(self.var_stop)
        elif var_name == 'stop_mip':
            # var_stop_mip contains already the solution values
            res = pd.DataFrame.from_dict(self.var_stop_mip, orient='index')
        elif var_name == 'charge':
            res = self.get_var_solution_as_df(self.var_charge)
        elif var_name == 'discharge':
            res = self.get_var_solution_as_df(self.var_discharge)
        elif var_name == 'exc':
            res = self.get_var_solution_as_df(self.var_exc)
        elif var_name == 'prices':
            res = pd.DataFrame()
            for c in self.solver.constraints():
                if 'balance' in c.name():
                    if self.idx_ts_type == pd.Timestamp:
                        timestamp = pd.Timestamp(c.name().split('_')[1])
                    else:
                        timestamp = c.name().split('_')[1]
                    area = c.name().split('_')[-1]
                    res.loc[timestamp, area] = c.dual_value()
        else:
            raise ValueError(f'Unknown variable name: {var_name}.')

        return res

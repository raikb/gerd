'''Contains input checking for input of Dispatch in models.'''
import logging


# Logging
logger = logging.getLogger(__name__)

# %% Definition of input format and others
_mandatory_input_columns = {
    'generators': ['area', 'p_min', 'p_max'],
    'storages': ['area', 'cap_ini', 'cap_max', 'efficiency', 'power'],
    'ntcs': ['area1', 'area2', 'ntc'],
    'var_costs': [],
    'load': []
    }

_index_names = {'generators': 'name', 'storages': 'name',
                'load': 'time', 'ntcs': 'time', 'var_costs': 'time'}

_model_types = ['mip', 'rmip', 'mip_rmip']


# %% Check and adjust input format/data

def check_mandatory_input_columns(df, input_type: str):
    '''Raises a ValueError if the input DataFrame does not contain all
    mandatory input columns. No error is raised if there are
    additional columns so that the user can define columns
    for other purposes.
    :param df: Input DataFrame
    :param input_type: Input type as string'''

    is_included = set(_mandatory_input_columns[input_type]).issubset(
        set(df.columns))
    if not is_included:
        raise ValueError(f'Columns of the input type {input_type} should be '
                         f'{_mandatory_input_columns[input_type]}')


def check_index_name(df, input_type: str):
    '''Looks for columns that might contain the index and sets it,
    if the index name is not correct. A ValueError is raised,
    if the index is not or cannot be set. If all is fine, df is returned.
    The index is important because it is used as a key for
    the variables in the optimization.

    :param df: Input DataFrame
    :param input_type: Input type as string
    :rtype: pd.DataFrame'''

    if not _index_names[input_type] == df.index.name:
        if _index_names[input_type] in df.columns:
            df = df.set_index(_index_names[input_type])
            return df
        else:
            raise ValueError(f'The index for {input_type} input should be '
                             f'{_index_names[input_type]}')
    else:
        return df


def check_duplicates_df_index(df):
    '''Checks if there are duplicated in a DataFrame's index.

    :param df: List of input DataFrames or a single input DataFrame'''

    if not isinstance(df, list):
        df = [df]

    for dfi in df:
        if dfi.index.duplicated().any():
            duplicates = dfi.index[dfi.index.duplicated()].tolist()
            raise ValueError('Duplicated names are not allowed. Please check: '
                             f'{", ".join(str(x) for x in duplicates)}')


def create_area_name_index(df):
    '''Converts the index of the input DataFrame into a "area_name" string.

    :param df: pandas DataFrame
    :rtype: pd.DataFrame'''

    df = df.set_index(df['area'] + '_' + df.index)
    return df


def check_model_type(model_type: str, model_generators: bool):
    '''Checks the solver type and raises an ValueError if undefined.
    The model type is changed to rmip if a MIP is defined but no generation
    modeling.

    :param model_type: Solver type'''

    if model_type not in _model_types:
        raise ValueError(f'Unknown solver type.'
                         'Please choose among: {_model_types}')

    is_mip = ((model_type == 'mip') or (model_type == 'mip_rmip'))

    if is_mip and not model_generators:
        model_type = 'rmip'
        logger.warning('Solving a MIP without modeling generators is '
                       'not allowed. The model type is changed to RMIP.')

    return model_type


# %% Check generator specific input

def check_generators_power_bounds(df):
    '''Maximum power should be greater than minimum power. Otherwise,
    a ValueError is raised.
    :param df: Input DataFrame for generators'''

    if any(df['p_min'] >= df['p_max']):
        raise ValueError('Maximum power of generators '
                         f'{df[df["p_min"] >= df["p_max"]].index.to_list()}'
                         ' should be greater than minimum power')


# %% Check storage specific input

def check_storage_cap_ini_max(df):
    '''Initial capacity cannot be greater than the maximum capacity.
    Otherwise, a ValueError is raised.'''

    if any(df['cap_ini'] > df['cap_max']):
        raise ValueError('Initial capacity of storages '
                         f'{df[df["cap_ini"] > df["cap_max"]].index.to_list()}'
                         ' is greater than the maximum capacity')


def check_storage_efficiency(df):
    '''Efficiency should be between 0 and 1.
    Otherwise, a ValueError is raised.'''

    if any(df['efficiency'] > 1) or any(df['efficiency'] <= 0):
        raise ValueError('Storage efficiencies should be in '
                         'the interval [0,1]')


# %% Convenient functions for checking input DataFrames

def check_input_df(input_data, input_type):
    '''Checks format and some values of an input DataFrame in input_data.'''

    df = input_data[input_type].copy()

    # Check if all columns are included
    check_mandatory_input_columns(df, input_type)

    # Check and adjust index
    df = check_index_name(df, input_type)
    if input_type in ['storages', 'generators']:
        df = create_area_name_index(df)
        df.index.name = _index_names[input_type]

    # Sort time series data
    if df.index.name == 'time':
        df = df.sort_index()

    # Check input values
    if input_type == 'storages':
        check_storage_cap_ini_max(df)
        check_storage_efficiency(df)
    elif input_type == 'generators':
        check_generators_power_bounds(df)

    return df

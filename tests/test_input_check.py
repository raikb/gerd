'''Tests for input_check.
Classes are used to structure tests by functions.'''
import pandas as pd
import pytest
import re

from context import input_check


class Test_check_mandatory_input_columns():

    @pytest.mark.parametrize('input_type', ['generators', 'storages'])
    def test_nonexisting_mandatory_input_column(
            self, dispatch_input_one_area, input_type):
        '''ValueError is expected if one mandatory column does not exist.
        '''
        expected_msg = f'Columns of the input type {input_type} should be'
        df = dispatch_input_one_area[input_type]

        del df[input_check._mandatory_input_columns[input_type][0]]

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_mandatory_input_columns(df, input_type)

    @pytest.mark.parametrize('input_type', ['generators', 'storages'])
    def test_additional_input_column(
            self, dispatch_input_one_area, input_type):
        '''None should be returned if all mandatory input columns exist even
        if there is an "unknown" additional input column.
        '''
        df = dispatch_input_one_area[input_type]
        df['unknown'] = 0

        assert input_check.check_mandatory_input_columns(
            df, input_type) is None


class Test_check_index_name():

    @pytest.mark.parametrize(
        'input_type', ['generators', 'storages', 'load', 'var_costs'])
    def test_wrong_index_name(self, dispatch_input_one_area, input_type):
        '''ValueError is expected if the index has the wrong name.'''

        expected_msg = f'The index for {input_type} input should be '
        df = dispatch_input_one_area[input_type]
        df.index.name = 'wrong_name'

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_index_name(df, input_type)

    @pytest.mark.parametrize(
        'input_type', ['generators', 'storages', 'load', 'var_costs'])
    def test_correct_index_name(self, dispatch_input_one_area, input_type):
        '''df should be returned if the index name is correct'''
        input_df = dispatch_input_one_area[input_type]
        output_df = input_check.check_index_name(input_df, input_type)

        assert input_df.equals(output_df)

    @pytest.mark.parametrize(
        'input_type', ['generators', 'storages', 'load', 'var_costs'])
    def test_finding_index_in_columns(
            self, dispatch_input_one_area, input_type):
        '''check_index_name() should set the index if it is a column.'''

        input_df = dispatch_input_one_area[input_type]
        input_df = input_df.reset_index()
        input_df = input_check.check_index_name(input_df, input_type)

        assert input_df.index.name == input_check._index_names[input_type]


class Test_check_duplicates_df_index():

    def test_check_duplicates_df_index4df(self):
        '''ValueError should be raised if the index of a DataFrame contains
        duplicates. Duplicates should be contained in the error message.
        '''
        expected_msg = 'Please check: nuc, coal'
        df = pd.DataFrame(index=['nuc', 'nuc', 'coal', 'coal', 'gas'])

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_duplicates_df_index(df)

    def test_check_duplicates_df_index4dfs(self):
        '''ValueError should be raised if the index of at least one DataFrame
        in a list of DataFrames contains duplicates. Duplicates should be
        contained in the error message.
        '''
        expected_msg = 'Please check: pump1'
        df1 = pd.DataFrame(index=['lignite', 'gas'])
        df2 = pd.DataFrame(index=['pump1', 'pump1', 'pump2'])
        df3 = pd.DataFrame(index=['nuc', 'nuc', 'coal', 'coal', 'gas'])
        df = [df1, df2, df3]

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_duplicates_df_index(df)


class Test_check_model_type():

    @pytest.mark.parametrize('model_generators', [True, False])
    def test_unknown_solver_type(self, model_generators):
        '''ValueError is expected if the solver type is unknown.'''

        expected_msg = 'Unknown solver type.'

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_model_type('mip_mip', model_generators)

    def test_mip_wio_generators(self):
        '''Model type should be set to mip if a mip is chosen without
        modeling generators.'''

        assert input_check.check_model_type('mip', False) == 'mip'
        assert input_check.check_model_type('mip_rmip', False) == 'mip'


class Test_check_generators_power_bounds():

    def test_generator_p_min_max(self, dispatch_input_one_area):
        '''ValueError is expected if minimum power is greater
        than maximum power.'''

        expected_msg = 'should be greater than minimum power'

        input_generators = dispatch_input_one_area['generators']
        input_generators['p_max'] = 1
        input_generators['p_min'] = 1

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_generators_power_bounds(input_generators)


class Test_check_storage_cap_ini_max():

    def test_storage_input_cap_ini(self, dispatch_input_one_area):
        '''ValueError is expected if the initial capacity is greater than
        the maximum capacity of the storage.
        '''
        expected_msg = 'is greater than the maximum capacity'

        input_storages = dispatch_input_one_area['storages']
        input_storages['cap_ini'] = 2
        input_storages['cap_max'] = 1

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_storage_cap_ini_max(input_storages)


class Test_check_storage_efficiency():

    @pytest.mark.parametrize('efficiency', [0, 2])
    def test_storage_efficiency(self, dispatch_input_one_area, efficiency):
        '''ValueError is expected if the initial capacity is greater than
        the maximum capacity of the storage.
        '''
        expected_msg = 'Storage efficiencies should be in the interval [0,1]'
        input_storages = dispatch_input_one_area['storages']
        input_storages['efficiency'] = efficiency

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_check.check_storage_efficiency(input_storages)


class Test_check_input_df():

    @pytest.mark.parametrize(
        'input_type', ['generators', 'storages'])
    def test_check_generator_input(self, dispatch_input_one_area, input_type):
        '''Test with one area. The index name should be set to time and
        the index should be defined as area-name combination and hence
        different from the input index.'''

        input_df = dispatch_input_one_area[input_type]
        output_df = input_check.check_input_df(
            dispatch_input_one_area, input_type)
        assert output_df.index.name == 'name'
        assert all(input_df.index != output_df.index)

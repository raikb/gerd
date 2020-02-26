'''This file contains tests for the Dispatch model and tests
indirectly the class ModelSolver as well.
It uses only the fixtures from conftest.py.
Classes are used to structure tests by content. Test are written and named
to make them self-explanatory.'''
import pytest
import re

from context import models


class TestPrices:
    '''Testing whether the model produces prices as expected for
    the different test cases. Rounding is advised because of minor
    numerical deviations.'''

    @pytest.mark.parametrize(
        "t,price", [['t01', 15], ['t02', 30], ['t03', 30], ['t04', 15]])
    def test_one_area_prices_wo_storage(
            self, dispatch_input_one_area, t, price):
        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area, model_storages=False)
        dispatch.optimize()

        assert round(dispatch.solution['prices'].loc[t, 'de']) == price

    @pytest.mark.parametrize(
        "iloc_begin,iloc_end,price",
        [[0, 500, 15], [501, 1000, 30]])
    def test_one_area_long_prices(
            self, dispatch_input_one_area_long, iloc_begin, iloc_end, price):
        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area_long, model_storages=False)
        dispatch.optimize(rolling_optimization=True)

        assert all(round(
            dispatch.solution[
                'prices']['de'].iloc[iloc_begin:iloc_end]) == price)

    @pytest.mark.parametrize(
        "t,model_type,price",
        [['t01', 'rmip', 15], ['t02', 'rmip', 30],
         ['t03', 'rmip', 30], ['t04', 'rmip', 15],
         ['t01', 'mip_rmip', 15], ['t02', 'mip_rmip', 30],
         ['t03', 'mip_rmip', 30], ['t04', 'mip_rmip', 15]])
    def test_rmip_vs_mip_rmip(
            self, dispatch_input_one_area, model_type, t, price):
        '''Prices should be the same for both model types because RMIP
        is running at the end.'''
        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area, model_storages=False)
        dispatch.optimize(model_type)

        assert round(dispatch.solution['prices'].loc[t, 'de']) == price

    @pytest.mark.parametrize(
        "t,area,price", [['t01', 'de', 5], ['t02', 'de', 15],
                         ['t01', 'at', 5], ['t02', 'at', 15],
                         ['t01', 'fr', 5], ['t02', 'fr', 5]])
    def test_three_areas_prices(
            self, dispatch_input_three_areas, t, area, price):
        dispatch = models.Dispatch(
            input_data=dispatch_input_three_areas)
        dispatch.optimize()

        assert round(dispatch.solution['prices'].loc[t, area]) == price


class TestCrossBorderFlows():

    @pytest.mark.parametrize(
        "t,area1,area2,flow", [['t01', 'de', 'at', 100],
                               ['t02', 'de', 'at', 100],
                               ['t01', 'de', 'fr', -500],
                               ['t02', 'de', 'fr', -100]])
    def test_flows(
            self, dispatch_input_three_areas, t, area1, area2, flow):
        dispatch = models.Dispatch(
            input_data=dispatch_input_three_areas)
        dispatch.optimize()

        assert round(dispatch.solution['exc'].loc[t, (area1, area2)]) == flow

    def test_switch_off_ntcs(self, dispatch_input_three_areas):
        expected_msg = 'NTC input data defined but model_exchange'

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            models.Dispatch(
                input_data=dispatch_input_three_areas, model_exchange=False)

    def test_force_ntcs(self, dispatch_input_one_area):
        '''Even forced NTC modeling should be ignored if there is no
        respective input data.'''

        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area,
            model_exchange=True)
        dispatch.optimize()
        assert dispatch.model_exchange is False


class TestInputProcessing():

    def test_disable_generators_and_storages(self, dispatch_input_one_area):
        expected_msg = 'Either storages or generators have to be defined'

        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            models.Dispatch(
                input_data=dispatch_input_one_area,
                model_generators=False, model_storages=False)

    def test_misspell_column_min_dn_time(self, dispatch_input_one_area):
        dispatch_input_one_area['generators'] = dispatch_input_one_area[
            'generators'].rename(columns={'min_dn_time': 'wrong_name'})
        dispatch = models.Dispatch(input_data=dispatch_input_one_area)
        dispatch.optimize()
        assert dispatch.model_min_dn_time is False


class TestEnergyBalance():

    @pytest.mark.parametrize(
        "t,load", [['t01', 400], ['t02', 1100], ['t03', 1000], ['t04', 400]])
    def test_energy_balance(self, dispatch_input_one_area, t, load):
        dispatch = models.Dispatch(input_data=dispatch_input_one_area)
        dispatch.optimize()

        prod = dispatch.solution['prod'].loc[t].sum()
        charge = dispatch.solution['charge'].loc[t].sum()
        discharge = dispatch.solution['discharge'].loc[t].sum()

        assert (prod - charge + discharge) == load


class TestProduction():

    def test_must_run(self, dispatch_input_one_area):
        dispatch_input_one_area['generators'].loc['gas', 'p_must_run'] = 400
        dispatch = models.Dispatch(input_data=dispatch_input_one_area)
        dispatch.optimize()
        for t in dispatch.idx_ts:
            prod = dispatch.solution['prod'].loc[t, 'de_gas']
            on = dispatch.solution['on'].loc[t, 'de_gas']
            assert prod >= (400 * on)

    @pytest.mark.parametrize(
        "t,name,production,on",
        [['t01', 'de_coal', 400, 1], ['t02', 'de_coal', 800, 1],
         ['t03', 'de_coal', 800, 1], ['t04', 'de_coal', 400, 1],
         ['t01', 'de_gas', 0, 0], ['t02', 'de_gas', 300, 1],
         ['t03', 'de_gas', 200, 1], ['t04', 'de_gas', 0, 0]])
    def test_mip(self, dispatch_input_one_area, t, name, production, on):
        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area, model_storages=False)
        dispatch.optimize('mip')

        assert round(dispatch.solution['prod'].loc[t, name]) == production
        assert dispatch.solution['on'].loc[t, name] == on


class TestStorages():

    @pytest.mark.parametrize(
        "t,charge,discharge",
        [['t01', 0, 10], ['t02', 20, 0], ['t03', 0, 50], ['t04', 0, 50]])
    def test_charging_discharging(
            self, dispatch_input_only_storage, t, charge, discharge):
        dispatch = models.Dispatch(
            input_data=dispatch_input_only_storage)
        dispatch.optimize('rmip')

        assert dispatch.solution['charge'].loc[t, 'de_pump1'] == charge
        assert dispatch.solution['discharge'].loc[t, 'de_pump1'] == discharge

    def test_long_test_case_single(
            self, dispatch_input_only_storage_long):
        '''The storage should be empty at the end of this test case.'''

        dispatch = models.Dispatch(
            input_data=dispatch_input_only_storage_long)
        dispatch.optimize('rmip', rolling_optimization=False)

        assert dispatch.solution['storage_cap'].iloc[-1].values == 0
        assert all(dispatch.solution['charge'].iloc[::2] == 10)
        assert all(dispatch.solution['discharge'].iloc[1::2] == 10)

    def test_long_test_case_rolling(
            self, dispatch_input_only_storage_long):
        '''Should yield same results as  test_long_test_case_single()'''

        dispatch = models.Dispatch(
            input_data=dispatch_input_only_storage_long)
        dispatch.optimize('rmip', rolling_optimization=True)

        assert dispatch.solution['storage_cap'].iloc[-1].values == 0
        assert all(dispatch.solution['charge'].iloc[::2] == 10)
        assert all(dispatch.solution['discharge'].iloc[1::2] == 10)


class TestRollingOptimization():

    def test_setting_rolling_opt_parameters(
            self, dispatch_input_one_area_long):
        nr_opt_horizons = 4
        nr_cut_off_steps = 20
        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area_long, model_storages=False)
        dispatch.optimize(
            rolling_optimization=True,
            nr_opt_horizons=nr_opt_horizons,
            nr_cut_off_steps=nr_cut_off_steps)

        assert dispatch.nr_opt_horizons == nr_opt_horizons
        assert dispatch.nr_cut_off_steps == nr_cut_off_steps

    def test_too_small_data_set(self, dispatch_input_one_area):
        '''Should optimize in one step even if rolling is activated.'''

        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area, model_storages=False)
        dispatch.optimize(
            rolling_optimization=True, nr_opt_horizons=4, nr_cut_off_steps=20)

        assert hasattr(dispatch, 'nr_opt_horizons') is False
        assert hasattr(dispatch, 'nr_cut_off_steps') is False

    def test_min_up_time(
            self, dispatch_input_one_area_up_time):
        '''The more expensive gas-fired power plant is forced to
        stay online for 150 hours.
        "model_min_dn_time=False" not need but it increases test coverage.'''

        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area_up_time,
            model_min_dn_time=False)
        dispatch.optimize(rolling_optimization=True)

        assert all(dispatch.solution['on']['de_coal'] > 0) is True
        assert sum(dispatch.solution['on']['de_gas'] > 0) == 150

    def test_min_dn_time(
            self, dispatch_input_one_area_dn_time):
        '''de_gas_long_dn_time is forced to shut down and stays offline
        because of the long minimum down time even though it is
        cheaper than the other gas-fired power plant.
        "model_min_up_time=False" not need but it increases test coverage.'''

        dispatch = models.Dispatch(
            input_data=dispatch_input_one_area_dn_time,
            model_min_up_time=False)
        dispatch.optimize('mip', rolling_optimization=True)

        assert sum(dispatch.solution['on']['de_gas_long_dn_time'] == 0) == 100

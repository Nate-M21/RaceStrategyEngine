import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from RaceStrategyEngine.driver import Driver
from RaceStrategyEngine.tyre_model import TyreModel
from RaceStrategyEngine.race_configuration import RaceConfiguration
from RaceStrategyEngine.combinatorics import StrategyCombinatorics
from sympy import symbols, Piecewise
from unittest.mock import patch

x = symbols('x')

points_mapping = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}  # this is per current regs

# Define the overtake chances function using sympy.Piecewise

overtake_chances_piecewise = Piecewise(
    (0.05, x < 0.2),
    (0.1, x < 0.5),
    (0.15, x < 0.8),
    (0.4, x < 1),
    (0.98, x < 1.2),
    (0.99, True)
)


@pytest.fixture
def basic_race_config():
    return RaceConfiguration(
        points_distribution=points_mapping,
        pit_lane_time_loss=24,
        full_course_yellow_pitloss=None,
        probability_of_full_course_yellow=None,
        num_laps=50,
        total_fuel=100,
        fuel_effect_seconds_per_kg=0.033,
        fuel_consumption_per_lap=1.79,
        drs_boost=1,
        drs_activation_lap=3,
        time_lost_due_to_being_overtaken=0.6,
        time_lost_performing_overtake=0.1,
        min_time_lost_due_to_failed_overtake_attempt=0.1,
        max_time_lost_due_to_failed_overtake_attempt=0.6,
        overtake_chances_piecewise=overtake_chances_piecewise,
        race_start_stationary_time_penalty=2.5,
        race_start_grid_position_time_penalty=1.0,
    )


@pytest.fixture
def basic_tyre_models():
    x = symbols('x')
    return {
        "soft": TyreModel(90 + 0.1 * x),
        "medium": TyreModel(91 + 0.05 * x),
        "hard": TyreModel(92 + 0.02 * x)
    }


@pytest.fixture
def basic_driver_consistency():
    return {
        'mean': 0.15,
        'std_dev': 0.1,
        'min_lap_time_variation': 0.05,
        'mean_tyre_change_time': 2.3,
        'std_dev_tyre_change_time': 0.2
    }


@pytest.fixture
def basic_driver(basic_tyre_models, basic_driver_consistency, basic_race_config):
    return Driver('TestDriver', basic_tyre_models, basic_driver_consistency,
                  number_of_stops=2, race_configuration=basic_race_config)


def test_driver_initialization(basic_driver):
    assert basic_driver.name == 'TestDriver'
    assert basic_driver.num_of_stops == 2
    assert isinstance(basic_driver.strategy, StrategyCombinatorics)
    assert isinstance(basic_driver.selected_strategy, tuple)
    assert len(basic_driver._lambdified_tyre_models) == 3
    assert basic_driver.lap_time_mean_deviation == 0.15
    assert basic_driver.lap_time_std_dev == 0.1
    assert basic_driver.min_lap_time_variation == 0.05
    assert basic_driver.mean_tyre_change_time == 2.3
    assert basic_driver.std_dev_tyre_change_time == 0.2


def test_available_tyres_constraint(basic_driver):
    assert basic_driver.available_tyres_constraint is None


def test_available_compounds(basic_driver):
    assert basic_driver.available_compounds is None


def test_laps_used(basic_driver):
    assert basic_driver.laps_used is None


def test_strategy_options(basic_driver):
    assert isinstance(basic_driver.strategy_options, pd.DataFrame)
    assert not basic_driver.strategy_options.empty


def test_selected_strategy_setter(basic_driver):
    new_strategy = (3600, ('soft', 20), ('medium', 30))
    basic_driver.selected_strategy = new_strategy
    assert basic_driver.selected_strategy == new_strategy

    with pytest.raises(ValueError):
        basic_driver.selected_strategy = (3600, ('soft', 50))  # Only one compound


def test_get_alternate_strategies(basic_driver):
    strategies = basic_driver.get_alternate_strategies()
    assert isinstance(strategies, list)
    assert len(strategies) > 0

    dict_strategies = basic_driver.get_alternate_strategies(dict_mode=True)
    assert isinstance(dict_strategies, dict)


def test_highest_amount_of_stops_in_alternate_strategies(basic_driver):
    assert basic_driver.highest_amount_of_stops_in_alternate_strategies == 2


def test_lambdified_tyre_models(basic_driver):
    assert isinstance(basic_driver.lambdified_tyre_models, dict)
    assert len(basic_driver.lambdified_tyre_models) == 3
    assert all(callable(model) for model in basic_driver.lambdified_tyre_models.values())


def test_tyre_models(basic_driver, basic_tyre_models):
    assert basic_driver.tyre_models == basic_tyre_models


def test_num_of_stops(basic_driver):
    assert basic_driver.num_of_stops == 2


def test_race_config(basic_driver, basic_race_config):
    assert basic_driver.race_config == basic_race_config


def test_update_tyre_usage(basic_driver):
    basic_driver.update_tyre_usage([('soft', 10)], 'medium')
    assert basic_driver.strategy.used_tyres == [('soft', 10)]
    assert 'soft' in basic_driver.selected_strategy[1][0]


def test_update_tyre_models(basic_driver):
    new_models = {
        "soft": TyreModel(89 + 0.12 * symbols('x')),
        "medium": TyreModel(90 + 0.06 * symbols('x')),
        "hard": TyreModel(91 + 0.03 * symbols('x'))
    }
    basic_driver.update_tyre_models(new_models)
    assert basic_driver.tyre_models == new_models


def test_cull_tyre(basic_driver):
    basic_driver.cull_tyre('soft')
    assert 'soft' not in basic_driver.tyre_models
    assert all('soft' not in strategy for strategy in basic_driver.get_alternate_strategies())

def test_user_generated_strategy(basic_driver):
    # Test setting a user-generated strategy without race time
    lap_distribution = (('soft', 20), ('medium', 30))
    user_strategy = basic_driver.strategy_creator(lap_distribution)
    basic_driver.selected_strategy = user_strategy
    assert basic_driver.selected_strategy[0] == "User generated strategy (No theoretical clean air race time)"
    assert basic_driver.selected_strategy[1:] == user_strategy[1:]

    # Test setting a user-generated strategy with custom race time string
    custom_strategy = ("User generated strategy (Based on team simulation)", ('soft', 20), ('medium', 30))
    basic_driver.selected_strategy = custom_strategy
    assert basic_driver.selected_strategy == custom_strategy

    # Test setting a strategy with numerical race time
    numerical_strategy = (3600.5, ('soft', 20), ('medium', 30))
    basic_driver.selected_strategy = numerical_strategy
    assert basic_driver.selected_strategy == numerical_strategy

    # Test invalid race time string
    with pytest.raises(ValueError):
        basic_driver.selected_strategy = ("Invalid strategy", ('soft', 20), ('medium', 30))

def test_validate_race_strategy(basic_driver):
    valid_strategy = (3600, ('soft', 20), ('medium', 30))
    basic_driver._validate_race_strategy(valid_strategy)  # Should not raise

    valid_user_strategy = ("User generated strategy ", ('soft', 20), ('medium', 30))
    basic_driver._validate_race_strategy(valid_user_strategy)  # Should not raise

    with pytest.raises(ValueError):
        basic_driver._validate_race_strategy((3600, ('soft', 50)))  # Only one compound

    with pytest.raises(ValueError):
        basic_driver._validate_race_strategy((3600, ('soft', 20), ('medium', 20)))  # Not enough laps

        # Invalid race times
        with pytest.raises(ValueError, match="Race time must be a positive number"):
            basic_driver._validate_race_strategy((0, ('soft', 25), ('medium', 25)))

        with pytest.raises(ValueError, match="Race time must be a positive number"):
            basic_driver._validate_race_strategy((-3600, ('soft', 25), ('medium', 25)))

        with pytest.raises(ValueError, match="Race time string must start with 'User generated strategy'"):
            basic_driver._validate_race_strategy(("Invalid strategy", ('soft', 25), ('medium', 25)))

        with pytest.raises(ValueError,
                           match="Race time must be either a positive number or a string starting with 'User generated strategy'"):
            basic_driver._validate_race_strategy(([], ('soft', 25), ('medium', 25)))

        # Invalid stint formats
        with pytest.raises(ValueError, match="Stints must be tuples of \(compound, laps\)"):
            basic_driver._validate_race_strategy((3600, ['soft', 25], ('medium', 25)))

        with pytest.raises(ValueError, match="Stints must be tuples of \(compound, laps\)"):
            basic_driver._validate_race_strategy((3600, ('soft', 25, 'extra'), ('medium', 25)))

        # Invalid compound names or lap counts
        with pytest.raises(ValueError,
                           match="Each stint must have a string compound that is valid compound name and positive number of laps"):
            basic_driver._validate_race_strategy((3600, ('ultra_soft', 25), ('medium', 25)))

        with pytest.raises(ValueError,
                           match="Each stint must have a string compound that is valid compound name and positive number of laps"):
            basic_driver._validate_race_strategy((3600, ('soft', -5), ('medium', 25)))

        with pytest.raises(ValueError,
                           match="Each stint must have a string compound that is valid compound name and positive number of laps"):
            basic_driver._validate_race_strategy((3600, ('soft', 'twenty-five'), ('medium', 25)))

        # Not enough different compounds
        with pytest.raises(ValueError, match="Strategy must use at least two different tyre compounds"):
            basic_driver._validate_race_strategy((3600, ('soft', 25), ('soft', 25)))

        # Total laps don't match race distance
        with pytest.raises(ValueError, match="Total laps in strategy .* must equal the total in the race"):
            basic_driver._validate_race_strategy((3600, ('soft', 20), ('medium', 20)))

        with pytest.raises(ValueError, match="Total laps in strategy .* must equal the total in the race"):
            basic_driver._validate_race_strategy((3600, ('soft', 30), ('medium', 30)))

        # Edge case: very small discrepancy in total laps (should pass)
        basic_driver._validate_race_strategy((3600, ('soft', 25.05), ('medium', 24.95)))  # Should not raise


def test_validate_stops_depth_limit(basic_driver):
    assert basic_driver._validate_stops_depth_limit(None) == 2
    assert basic_driver._validate_stops_depth_limit(1) == 1

    with pytest.raises(ValueError):
        basic_driver._validate_stops_depth_limit(3)  # More than num_of_stops


def test_validate_alternate_strategies_method():
    assert Driver._validate_alternate_strategies_method(None) == 'breadth'
    assert Driver._validate_alternate_strategies_method('depth') == 'depth'

    with pytest.raises(ValueError):
        Driver._validate_alternate_strategies_method('invalid')


def test_validate_top_n_strategies():
    assert Driver._validate_top_n_strategies(5) == 5

    with pytest.raises(ValueError):
        Driver._validate_top_n_strategies(0)


def test_get_top_n_fastest_strategies_per_stop(basic_driver):
    strategies = basic_driver._get_top_n_fastest_strategies_per_stop(top_n=2, max_stops=2)
    assert isinstance(strategies, list)
    assert len(strategies) <= 4  # 2 strategies for each of 1 and 2 stops

    dict_strategies = basic_driver._get_top_n_fastest_strategies_per_stop(top_n=2, max_stops=2, dict_mode=True)
    assert isinstance(dict_strategies, dict)
    assert len(dict_strategies) <= 2  # 1_stop and 2_stop keys


def test_driver_with_no_consistency_data(basic_tyre_models, basic_race_config):
    driver = Driver('NoConsistencyDriver', basic_tyre_models, {}, number_of_stops=2,
                    race_configuration=basic_race_config)
    assert driver.lap_time_mean_deviation == 0.25  # default value
    assert driver.lap_time_std_dev == 0.15  # default value
    assert driver.min_lap_time_variation == 0.05  # default value
    assert driver.mean_tyre_change_time == 2.5  # default value
    assert driver.std_dev_tyre_change_time == 0.3  # default value


def test_driver_with_partial_consistency_data(basic_tyre_models, basic_race_config):
    partial_consistency = {'mean': 0.1, 'std_dev': 0.05}
    driver = Driver('PartialConsistencyDriver', basic_tyre_models, partial_consistency, number_of_stops=2,
                    race_configuration=basic_race_config)
    assert driver.lap_time_mean_deviation == 0.1
    assert driver.lap_time_std_dev == 0.05
    assert driver.min_lap_time_variation == 0.05  # default value
    assert driver.mean_tyre_change_time == 2.5  # default value
    assert driver.std_dev_tyre_change_time == 0.3  # default value


def test_driver_with_invalid_race_config(basic_tyre_models, basic_driver_consistency):
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        Driver('InvalidConfigDriver', basic_tyre_models, basic_driver_consistency, number_of_stops=2,
               race_configuration={})


def test_driver_with_zero_stops(basic_tyre_models, basic_driver_consistency, basic_race_config):
    with pytest.raises(ValueError):
        Driver('ZeroStopsDriver', basic_tyre_models, basic_driver_consistency, number_of_stops=0,
               race_configuration=basic_race_config)


def test_driver_with_negative_stops(basic_tyre_models, basic_driver_consistency, basic_race_config):
    with pytest.raises(ValueError):
        Driver('NegativeStopsDriver', basic_tyre_models, basic_driver_consistency, number_of_stops=-1,
               race_configuration=basic_race_config)


def test_update_tyre_usage_with_invalid_compound(basic_driver):
    with pytest.raises(ValueError):
        basic_driver.update_tyre_usage([('ultra_soft', 10)], 'medium')


def test_update_tyre_usage_with_negative_laps(basic_driver):
    with pytest.raises(ValueError):
        basic_driver.update_tyre_usage([('soft', -5)], 'medium')


def test_cull_nonexistent_tyre(basic_driver):
    with pytest.raises(ValueError):
        basic_driver.cull_tyre('ultra_soft')


def test_selected_strategy_update_on_tyre_model_change(basic_driver, monkeypatch):
    def mock_best_strategy(self):
        return 3500, ('medium', 25), ('hard', 25)

    monkeypatch.setattr(StrategyCombinatorics, 'best_strategy', mock_best_strategy)

    new_models = {
        "medium": TyreModel(90 + 0.06 * symbols('x')),
        "hard": TyreModel(91 + 0.03 * symbols('x'))
    }
    basic_driver.update_tyre_models(new_models)
    assert basic_driver.selected_strategy == (3500, ('medium', 25), ('hard', 25))


@patch.object(StrategyCombinatorics, 'best_strategy')
def test2_selected_strategy_update_on_tyre_model_change(mock_best_strategy, basic_driver):
    mock_best_strategy.return_value = (3500, ('medium', 25), ('hard', 25))
    new_models = {
        "medium": TyreModel(90 + 0.06 * symbols('x')),
        "hard": TyreModel(91 + 0.03 * symbols('x'))
    }

    basic_driver.update_tyre_models(new_models)
    assert basic_driver.selected_strategy == (3500, ('medium', 25), ('hard', 25))
    mock_best_strategy.assert_called_once()


def test_get_alternate_strategies_with_invalid_method(basic_driver):
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        basic_driver._get_top_n_fastest_strategies_per_stop(method='invalid_method')


def test_get_alternate_strategies_with_excess_stops(basic_driver):
    with pytest.raises(ValueError):
        basic_driver._get_top_n_fastest_strategies_per_stop(max_stops=5)  # Assuming max stops is less than 5


def test_strategy_with_available_tyres_constraint(basic_tyre_models, basic_driver_consistency, basic_race_config):
    available_tyres = [('soft', 0), ('medium', 0), ('hard', 0)]
    driver = Driver('ConstrainedDriver', basic_tyre_models, basic_driver_consistency,
                    number_of_stops=2, race_configuration=basic_race_config,
                    available_tyres_constraint=available_tyres)
    assert driver.available_tyres_constraint == available_tyres
    assert driver.available_compounds == ('soft', 'medium', 'hard')
    assert driver.laps_used == (0, 0, 0)


def test_strategy_with_invalid_available_tyres(basic_tyre_models, basic_driver_consistency, basic_race_config):
    invalid_tyres = [('soft', 0), ('ultra_soft', 0)]  # 'ultra_soft' is not in basic_tyre_models
    with pytest.raises(ValueError):
        Driver('InvalidTyresDriver', basic_tyre_models, basic_driver_consistency,
               number_of_stops=2, race_configuration=basic_race_config,
               available_tyres_constraint=invalid_tyres)


def test_strategy_with_insufficient_available_tyres(basic_tyre_models, basic_driver_consistency, basic_race_config):
    insufficient_tyres = [('soft', 0)]  # Only one compound, which is not enough for a valid strategy
    with pytest.raises(ValueError):
        Driver('InsufficientTyresDriver', basic_tyre_models, basic_driver_consistency,
               number_of_stops=2, race_configuration=basic_race_config,
               available_tyres_constraint=insufficient_tyres)

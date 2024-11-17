import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from sympy import symbols
from RaceStrategyEngine.combinatorics import StrategyCombinatorics
from RaceStrategyEngine.tyre_model import TyreModel


@pytest.fixture
def basic_tyre_models():
    x = symbols('x')
    return {
        "soft": TyreModel(90 + 0.1 * x),
        "medium": TyreModel(91 + 0.05 * x),
        "hard": TyreModel(92 + 0.02 * x)
    }


@pytest.fixture
def basic_strategy_combinatorics(basic_tyre_models):
    return StrategyCombinatorics(basic_tyre_models, number_of_stops=2, num_laps=50, pit_loss=20)


def test_initialization(basic_strategy_combinatorics):
    assert basic_strategy_combinatorics.num_laps == 50
    assert basic_strategy_combinatorics.pit_loss == 20
    assert len(basic_strategy_combinatorics._tyre_models) == 3
    assert isinstance(basic_strategy_combinatorics._strategy_df, pd.DataFrame)


def test_strategy_df_property(basic_strategy_combinatorics):
    df = basic_strategy_combinatorics.strategy_df
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ['strategy', 'optimal_dist', 'racetime', 'stops'])


def test_available_tyres_constraint(basic_tyre_models):
    available_tyres = [("soft", 0), ("medium", 0), ("hard", 0)]
    sc = StrategyCombinatorics(basic_tyre_models, number_of_stops=2, num_laps=50, pit_loss=20,
                               available_tyres_constraint=available_tyres)
    assert sc.available_tyres_constraint == available_tyres
    assert sc.available_compounds == ("soft", "medium", "hard")
    assert sc.laps_used == (0, 0, 0)


def test_generate_combinations(basic_strategy_combinatorics):
    combinations = basic_strategy_combinatorics.generate_combinations()
    assert isinstance(combinations, list)
    assert all(isinstance(combo, tuple) for combo in combinations)
    assert all(len(set(combo)) > 1 for combo in combinations)  # Ensure at least two different compounds


def test_create_strategies_dataframe(basic_strategy_combinatorics):
    df = basic_strategy_combinatorics._create_strategies_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['strategy', 'optimal_dist', 'racetime', 'stops'])


def test_update_used_tyres_and_current_tyre(basic_strategy_combinatorics):
    basic_strategy_combinatorics.update_used_tyres_and_current_tyre([('soft', 10)], 'medium')
    assert basic_strategy_combinatorics.used_tyres == [('soft', 10)]
    assert all('soft' in strategy for strategy in basic_strategy_combinatorics._strategy_df['strategy'])


def test_cull_tyre(basic_strategy_combinatorics):
    initial_strategies = len(basic_strategy_combinatorics._strategy_df)
    basic_strategy_combinatorics.cull_tyre('soft')
    assert 'soft' not in basic_strategy_combinatorics._tyre_models
    assert len(basic_strategy_combinatorics._strategy_df) < initial_strategies


def test_best_strategy(basic_strategy_combinatorics):
    best_strategy = basic_strategy_combinatorics.best_strategy()
    assert isinstance(best_strategy, tuple)
    assert len(best_strategy) > 1
    assert isinstance(best_strategy[0], float)  # race time
    assert all(isinstance(stint, tuple) for stint in best_strategy[1:])  # (compound, laps) tuples


def test_find_n_lowest_strategies(basic_strategy_combinatorics):
    n_lowest = basic_strategy_combinatorics.find_n_lowest_strategies(2)
    assert isinstance(n_lowest, pd.DataFrame)
    assert len(n_lowest) == 2
    assert n_lowest.iloc[0]['racetime'] <= n_lowest.iloc[1]['racetime']


def test_filter_by_stop(basic_strategy_combinatorics):
    one_stop_strategies = basic_strategy_combinatorics.filter_by_stop(1)
    assert isinstance(one_stop_strategies, pd.DataFrame)
    assert all(one_stop_strategies['stops'] == 1)


def test_match_strategy_compounds_and_laps_used():
    strategy = ['soft', 'medium', 'hard']
    available = [('soft', 5), ('medium', 3), ('hard', 0), ('soft', 0), ('medium', 1)]

    result = StrategyCombinatorics.match_strategy_compounds_and_laps_used(strategy, available)
    assert result == [0, 1, 0]

    result = StrategyCombinatorics.match_strategy_compounds_and_laps_used(strategy, available, method='as_provided')
    assert result == [5, 3, 0]


def test_smart_round():
    numbers = [19.4, 24.3, 31.3]
    result = StrategyCombinatorics.smart_round(numbers, 75)
    assert sum(result) == 75
    assert all(isinstance(num, (int, np.integer)) for num in result)


def test_scale_values():
    numbers = [10, 20, 30]
    result = StrategyCombinatorics.scale_values(numbers, 120)
    assert sum(result) == pytest.approx(120)
    assert result[1] / result[0] == pytest.approx(2)
    assert result[2] / result[0] == pytest.approx(3)

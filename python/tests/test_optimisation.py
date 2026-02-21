import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import sympy as sp
from sympy import symbols
from RaceStrategyEngine.optimisation import StrategyOptimization
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
def basic_strategy_optimization(basic_tyre_models):
    return StrategyOptimization(basic_tyre_models, num_laps=50, pit_loss=20)


def test_initialization(basic_strategy_optimization):
    assert basic_strategy_optimization.num_laps == 50
    assert basic_strategy_optimization.pit_loss == 20
    assert len(basic_strategy_optimization._tyre_models) == 3


def test_tyre_model_initialization_error():
    with pytest.raises(TypeError, match="model must be a sympy expression"):
        # noinspection PyTypeChecker
        StrategyOptimization({"soft": TyreModel(90)}, num_laps=50, pit_loss=20)

def test_extract_common_symbol_with_errors(basic_tyre_models):
    so = StrategyOptimization(basic_tyre_models, num_laps=50, pit_loss=20)

    # Test with a valid model
    symbol = so.extract_tyre_model_symbol(basic_tyre_models["soft"].model)
    assert str(symbol) == 'x'

    # Test with a model containing no symbols
    with pytest.raises(ValueError, match="No symbols found in the model"):
        so.extract_tyre_model_symbol(sp.Integer(1))

    # Test with a model containing a non-Symbol type
    class FakeSymbol:
        pass

    fake_model = type('FakeExpr', (), {'free_symbols': {FakeSymbol()}})()
    with pytest.raises(TypeError, match="Expected type Symbol from sympy, got FakeSymbol instead"):
        so.extract_tyre_model_symbol(fake_model)

def test_different_symbols_for_different_models(basic_tyre_models):
    # valid model should have no errors
    valid = StrategyOptimization(basic_tyre_models, num_laps=50, pit_loss=20)

    x, y, z = symbols('x, y, z')
    invalid_tyres = {
        "soft": TyreModel(90 + 0.1 * x),
        "medium": TyreModel(91 + 0.05 * y),
        "hard": TyreModel(92 + 0.02 * z)
    }
    # invalid model should have value error
    with pytest.raises(ValueError):
        invalid = StrategyOptimization(invalid_tyres, num_laps=50, pit_loss=20)


def test_num_laps_setter(basic_tyre_models):
    so = StrategyOptimization(basic_tyre_models, num_laps=50, pit_loss=20)
    so.num_laps = 60
    assert so.num_laps == 60

    with pytest.raises(ValueError):
        so.num_laps = 0

    with pytest.raises(ValueError):
        so.num_laps = -10

    with pytest.raises(ValueError):
        so.num_laps = 1  # Should be at least 2

    with pytest.raises(ValueError):
        so.num_laps = float('inf')


def test_pit_loss_setter(basic_tyre_models):
    so = StrategyOptimization(basic_tyre_models, num_laps=50, pit_loss=20)
    so.pit_loss = 25
    assert so.pit_loss == 25

    so.pit_loss = 21.5
    assert so.pit_loss == 21.5

    with pytest.raises(ValueError):
        so.pit_loss = 0

    with pytest.raises(ValueError):
        so.pit_loss = -5

    with pytest.raises(ValueError):
        so.pit_loss = None

    with pytest.raises(ValueError):
        so.pit_loss = '20'

    with pytest.raises(ValueError):
        so.pit_loss = float('inf')


def test_used_tyres_setter(basic_strategy_optimization):
    basic_strategy_optimization.used_tyres = [("soft", 10), ("medium", 15)]
    assert basic_strategy_optimization.used_tyres == [("soft", 10), ("medium", 15)]

    with pytest.raises(ValueError):
        basic_strategy_optimization.used_tyres = [("soft", 0)]

    with pytest.raises(ValueError):
        basic_strategy_optimization.used_tyres = [("unknown", 10)]

    with pytest.raises(TypeError):
        basic_strategy_optimization.used_tyres = "not a list"


def test_strategy_comparison():
    required = ["soft", "medium", "hard"]
    proposed = ["soft", "medium", "hard", "soft"]

    # Test basic comparison
    is_valid, result = StrategyOptimization.strategy_comparison(required, proposed)
    assert is_valid
    assert result == ["soft", "medium", "hard", "soft"]

    # Test invalid strategy
    proposed = ["soft", "medium"]
    is_valid, result = StrategyOptimization.strategy_comparison(required, proposed)
    assert not is_valid
    assert result == ["soft", "medium"]  # The result should be the proposed strategy when not valid

    # Test with show_extras=True
    proposed = ["soft", "medium", "hard", "soft"]
    is_valid, result = StrategyOptimization.strategy_comparison(required, proposed, show_extras=True,
                                                                order_output=False)
    assert is_valid
    assert result == {"soft": 1}  # There's one extra "soft" compound

    # Test with order_output=True
    is_valid, result = StrategyOptimization.strategy_comparison(required, proposed, show_extras=False,
                                                                order_output=True)
    assert is_valid
    assert result == ["soft", "medium", "hard", "soft"]


def test_strategy_comparison_with_available_constraint():
    required = ["soft", "medium", "hard"]
    proposed = ["soft", "medium", "hard", "soft", "medium"]
    available = ["soft", "soft", "medium", "medium", "hard"]

    is_valid, result = StrategyOptimization.strategy_comparison(required, proposed,
                                                                available_strategy_constraint=available)
    assert is_valid
    assert result == ["soft", "medium", "hard", "soft", "medium"]

    # Test with more compounds in proposed than available
    proposed = ["soft", "soft", "soft", "medium", "hard"]
    is_valid, result = StrategyOptimization.strategy_comparison(required, proposed,
                                                                available_strategy_constraint=available)
    assert not is_valid
    assert result == ["soft", "soft", "soft", "medium", "hard"]


def test_optimal_lap_distribution(basic_strategy_optimization):
    result = basic_strategy_optimization.optimal_lap_distribution(["soft", "medium", "hard"])
    assert len(result) == 3
    assert sum(laps for _, laps in result) == pytest.approx(50, abs=1e-5)


def test_check_strategy_validity(basic_strategy_optimization):
    # Valid strategy
    basic_strategy_optimization._check_strategy_validity(["soft", "medium", "hard"])

    # Invalid strategies
    with pytest.raises(ValueError, match="Optimization requires at least two compounds"):
        basic_strategy_optimization._check_strategy_validity(["soft"])

    with pytest.raises(ValueError, match="Tyre compound\(s\) unknown not recognized"):
        basic_strategy_optimization._check_strategy_validity(["soft", "medium", "unknown"])

    # Test with used tyres
    basic_strategy_optimization.used_tyres = [("soft", 10)]
    with pytest.raises(ValueError, match="Invalid strategy: "):
        basic_strategy_optimization._check_strategy_validity(["medium", "hard"])


def test_validate_num_laps_and_stops():
    StrategyOptimization.validate_num_laps_and_stops(50, 2)  # Should not raise

    with pytest.raises(ValueError, match="Number of laps \(2\) must be at least 3"):
        StrategyOptimization.validate_num_laps_and_stops(2, 2)


def test_validate_pit_loss():
    StrategyOptimization.validate_pit_loss(20)  # Should not raise

    with pytest.raises(ValueError, match="Pit loss must be a positive number"):
        StrategyOptimization.validate_pit_loss(0)

    with pytest.raises(ValueError, match="Pit loss must be a positive number"):
        StrategyOptimization.validate_pit_loss(-5)


def test_extract_common_symbol(basic_tyre_models):
    so = StrategyOptimization(basic_tyre_models, num_laps=50, pit_loss=20)
    symbol = so.extract_tyre_model_symbol(basic_tyre_models["soft"].model)
    assert str(symbol) == 'x'


def test_generate_symbols(basic_tyre_models):
    so = StrategyOptimization(basic_tyre_models, num_laps=50, pit_loss=20)
    base_symbol = symbols('x')
    generated_symbols = so._generate_symbols(base_symbol, 4)  # For a 4-compound strategy
    assert len(generated_symbols) == 3
    assert str(generated_symbols[0]) == 'x1'
    assert str(generated_symbols[1]) == 'x2'
    assert str(generated_symbols[2]) == 'x3'

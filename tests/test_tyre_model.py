import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import sympy as sp
from RaceStrategyEngine.tyre_model import TyreModel


def test_tyre_model_creation():
    x = sp.Symbol('x')
    model = 90 + 0.1 * x
    tyre = TyreModel(model)
    assert isinstance(tyre.model, sp.Expr)
    assert tyre.max_laps == float('inf')


def test_tyre_model_with_max_laps():
    x = sp.Symbol('x')
    model = 90 + 0.1 * x
    tyre = TyreModel(model, max_laps=30)
    assert tyre.max_laps == 30


def test_tyre_model_invalid_max_laps():
    x = sp.Symbol('x')
    model = 90 + 0.1 * x
    with pytest.raises(ValueError):
        TyreModel(model, max_laps=0)
    with pytest.raises(ValueError):
        TyreModel(model, max_laps=-1)
    with pytest.raises(ValueError):
        TyreModel(model, max_laps=1.5)


def test_tyre_model_invalid_model():
    with pytest.raises(TypeError):
        TyreModel("not an expression")


def test_tyre_model_multiple_symbols():
    x, y = sp.symbols('x y')
    model = 90 + 0.1 * x + 0.2 * y
    with pytest.raises(ValueError):
        TyreModel(model)


def test_tyre_model_immutability():
    x = sp.Symbol('x')
    model = 90 + 0.1 * x
    tyre = TyreModel(model)
    with pytest.raises(AttributeError):
        # noinspection PyDataclass
        tyre.max_laps = 20


def test_tyre_model_repr():
    x = sp.Symbol('x')
    model = 90 + 0.1 * x
    tyre = TyreModel(model, max_laps=30)
    assert repr(tyre) == f"TyreModel(model={model}, max_laps=30)"


@pytest.mark.parametrize("max_laps,expected", [
    (float('inf'), float('inf')),
    (50, 50),
    (1, 1),
])
def test_tyre_model_max_laps_variations(max_laps, expected):
    x = sp.Symbol('x')
    model = 90 + 0.1 * x
    tyre = TyreModel(model, max_laps=max_laps)
    assert tyre.max_laps == expected

def test_tyre_model_no_variable():

    model = sp.Integer(21)

    with pytest.raises(ValueError):
        TyreModel(model)

def test_tyre_model_multiple_variables():
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    model = 90 + 2 * x + y**2

    with pytest.raises(ValueError):
        TyreModel(model)

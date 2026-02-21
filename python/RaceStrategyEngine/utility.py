"""This module defines the utility functions for various operations.

This module contains a collection of utility functions that can be used
across different parts of the project.

"""
import time
from dataclasses import dataclass
from functools import wraps
from typing import LiteralString, Optional, Any, TypedDict
from sympy import Piecewise, And, Or, Not, Eq, Gt, Lt, Ge, Le, Float, Integer
import pandas as pd


class DriverState(TypedDict):
    position: int
    total_time: float
    delta_to_leader: float
    used_tyres: list[tuple[str, int]]
    current_tyre: str
    current_tyre_laps_age: int

RaceState = dict[str, DriverState]


@dataclass
class RaceDataPacket:
    current_lap: int
    race_state: dict
    laptimes: dict



def safe_get_from_list(array: list, index: int, default: Optional[Any] = None):
    """
    Retrieves an element from a list at the specified index safely.

    This function attempts to retrieve the element at the specified index from the given list.
    If the index is out of range, it returns the default value instead of raising an IndexError.

    Args:
        array (list): The list from which to retrieve the element.
        index (int): The index of the element to retrieve.
        default (any, optional): The default value to return if the index is out of range. Defaults to None.

    Returns:
        any: The element at the specified index if it exists, or the default value if the index is out of range.
    """
    try:
        return array[index]
    except IndexError:
        return default

def get_threshold_value(condition):
    lhs = condition.lhs
    rhs = condition.rhs

    # Return whichever side is the number (not the symbol)
    if lhs.is_symbol:
        return float(rhs)  # x < 5 → return 5
    else:
        return float(lhs)  # 5 < x → return 5

def format_overtake_piecewise(overtake_piecewise_func: Piecewise) -> str:
    """
    Formats a Sympy Piecewise function into a readable string representation.

    Args:
        overtake_piecewise_func (Piecewise): The Sympy Piecewise function to format.

    Returns:
        str: A formatted string representation of the Piecewise function.
    """
    term_for_difference = 'Pace differential'

    def _format_condition(condition) -> str | LiteralString:
        match condition:
            case Lt() | Le() | Gt() | Ge() | Eq():
                return _format_comparison(condition)
            case And():
                return " and ".join(_format_condition(arg) for arg in condition.args)
            case Or():
                return " or ".join(_format_condition(arg) for arg in condition.args)
            case Not():
                return f"not ({_format_condition(condition.args[0])})"
            case _:
                return str(condition)

    def _format_comparison(comparison):
        _, rhs = comparison.args
        match comparison:
            case Lt():
                return f"{term_for_difference} < {rhs:.3f}s"
            case Le():
                return f"{term_for_difference} <= {rhs:.3f}s"
            case Gt():
                return f"{term_for_difference} > {rhs:.3f}s"
            case Ge():
                return f"{term_for_difference} >= {rhs:.3f}s"
            case Eq():
                return f"{term_for_difference} == {rhs:.3f}s"
            case _:
                return str(comparison)

    def _format_expr(expr):
        match expr:
            case Float() | Integer():
                return f"{float(expr):.2%}"
            case _:
                return str(expr)

    result = []
    for (probability, condition) in overtake_piecewise_func.args:
        if condition == True:  # The reason for (==True) is because that is 'else' case for Piecewise functions in sympy  # noqa: E712
            result.append(f"\t{term_for_difference}: any other case: overtake chance - {_format_expr(probability)}")
        else:
            condition_str = _format_condition(condition)
            result.append(f"\t{condition_str}: overtake chance - {_format_expr(probability)}")
    return "\n".join(result)


def time_simulation(func=None, *, message=None):
    """Decorator to primarily measure and print the execution time of the simulation methods

    This decorator can be used with or without arguments. It wraps the decorated
    function, measures its execution time, and prints the result. The function's
    metadata is preserved using the @wraps decorator.

    Args:
        func (callable, optional): The function to be decorated. Defaults to None.
        message (str, optional): A custom message to be printed with the execution time.
            Must be specified as a keyword argument. Defaults to None.

    Returns:
        callable: A wrapped version of the input function that prints execution time.
    
    Notes:
        Can be used to measure the execution time of any function or method

    Examples:
        @time_simulation
        def my_function():
            pass

        @time_simulation(message="Custom timing message")
        def another_function():
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            if message:
                print(f"{message} {execution_time:.6f} seconds")
            else:
                print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")

            return result

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def count_driver_simulations(pandas_df: pd.DataFrame, driver: str) -> int:
    """Count the number of simulations for a specific driver in a DataFrame.

    This function calculates the number of rows in the DataFrame that correspond
    to the specified driver. It's useful for determining the sample size of
    Monte Carlo simulations for a particular driver.

    Args:
        pandas_df (pd.DataFrame): A pandas DataFrame containing simulation results.
            Must include a 'driver' column.
        driver (str): The name of the driver to count simulations for.

    Returns:
        int: The number of simulations for the specified driver.

    Raises:
        ValueError: If the DataFrame does not contain a 'driver' column.
    """
    if 'driver' not in pandas_df.columns:
        raise ValueError("The DataFrame does not contain a 'driver' column")
    return len(pandas_df[pandas_df['driver'] == driver])


# driver colours and styles used for plotting around the project
driver_styles_for_plotting = {

    "Verstappen": {"color": "#FFFF00", "marker": "circle", 'line': {'dash': 'solid'}},
    "Perez": {"color": "#FFFF00", "marker": "star", 'line': {'dash': 'dash'}},
    "Hamilton": {"color": "#00D2BE", "marker": "circle", 'line': {'dash': 'solid'}},
    "Russell": {"color": "#00D2BE", "marker": "star", 'line': {'dash': 'dash'}},
    "Leclerc": {"color": "#DC0000", "marker": "circle", 'line': {'dash': 'solid'}},
    "Sainz": {"color": "#DC0000", "marker": "star", 'line': {'dash': 'dash'}},
    "Gasly": {"color": "#DDA0DD", "marker": "circle", 'line': {'dash': 'solid'}},
    "Ocon": {"color": "#DDA0DD", "marker": "star", 'line': {'dash': 'dash'}},
    "Norris": {"color": "#FF8700", "marker": "circle", 'line': {'dash': 'solid'}},
    "Piastri": {"color": "#FF8700", "marker": "star", 'line': {'dash': 'dash'}},
    "Alonso": {"color": "#00FF00", "marker": "circle", 'line': {'dash': 'solid'}},
    "Stroll": {"color": "#00FF00", "marker": "star", 'line': {'dash': 'dash'}},
    "Sargeant": {"color": "#005AFF", "marker": "circle", 'line': {'dash': 'solid'}},
    "Albon": {"color": "#005AFF", "marker": "star", 'line': {'dash': 'dash'}},
    "Zhou": {"color": "#900000", "marker": "circle", 'line': {'dash': 'solid'}},
    "Bottas": {"color": "#900000", "marker": "star", 'line': {'dash': 'dash'}},
    "Magnussen": {"color": "#F0F0F0", "marker": "circle", 'line': {'dash': 'solid'}},
    "Hulkenberg": {"color": "#F0F0F0", "marker": "star", 'line': {'dash': 'dash'}},
    "Ricciardo": {"color": "#4E7C9B", "marker": "circle", 'line': {'dash': 'solid'}},
    "Tsunoda": {"color": "#4E7C9B", "marker": "star", 'line': {'dash': 'dash'}},

}

def f1_radio(speaker, message, emoji="👨‍💻"):
    lines = message.strip().split('\n')
    max_length = max(len(line) for line in lines)
    width = max_length + 4
    
    print("╔" + "═" * width + "╗")
    print("║" + " " * width + "║")
    
    for line in lines:
        padding = width - len(line)
        print("║ " + line + " " * (padding - 1) + "║")
    
    print("║" + " " * width + "║")    
    print("╚" + "═" * width + "╝")
    print(f"   \\ -- {speaker}")
    print(f"     {emoji}")
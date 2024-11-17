"""This module defines the utility functions for various operations.

This module contains a collection of utility functions that can be used
across different parts of the project. 

"""
import sys
import time
from dataclasses import dataclass
from functools import wraps
from typing import LiteralString, Optional, Any
from sympy import Piecewise, And, Or, Not, Eq, Gt, Lt, Ge, Le, Float, Integer
from collections import namedtuple
import pandas as pd
import redis
import json
import msgspec


class SharedRedisResultStack:
    def __init__(self, key: str = 'redis_result_stack', host='localhost', port=6379, db=0, max_length=750):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.key = key
        self.max_length = max_length

    def append(self, result):
        if self.redis_client.llen(self.key) >= self.max_length:
            self.redis_client.lpop(self.key)
        self.redis_client.rpush(self.key, json.dumps(result))

    def pop(self):
        result = self.redis_client.rpop(self.key)
        return json.loads(result) if result else None

    def __len__(self):
        return self.redis_client.llen(self.key)


def connect_to_redis(redis_host, redis_port, redis_db):
    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        redis_client.ping()
        return redis_client
    except redis.exceptions.ConnectionError:
        print("\nUnable to connect to Redis. Please ensure the Redis server is running.")
        sys.exit()


RedisConnectionParameters = namedtuple('RedisConnectionParameters',
                                       ['redis_host', 'redis_port', 'redis_db', 'shared_state_db'])


@dataclass
class RaceDataPacket:
    current_lap: int
    race_state: dict
    laptimes: dict


class SharedRaceStateStore:
    def __init__(self, key: str = 'current_race_data', host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.key = key

    def update_shared_race_state(self, new_race_data: RaceDataPacket):
        if not isinstance(new_race_data, RaceDataPacket):
            raise TypeError(f'Expected RaceDataPacket, got {type(new_race_data).__name__}')

        serialized_race_data = msgspec.msgpack.encode(new_race_data)
        try:
            self.redis_client.set(self.key, serialized_race_data)
        except redis.RedisError as e:
            print(f"Error updating Redis: {e}")

    def get_shared_race_state(self) -> RaceDataPacket | None:

        try:
            current_serialized_race_data: bytes = self.redis_client.get(self.key)
            if current_serialized_race_data:
                current_race_data = msgspec.msgpack.decode(current_serialized_race_data, type=RaceDataPacket)
                return current_race_data
            else:
                return None
        except redis.RedisError as e:
            print(f"Error retrieving from Redis: {e}")


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


# I am using the '*' so that the message parameter should always be specified by name.
# Also using the @wraps decorator so the metadata of the actual function is maintained
# Both can be removed

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

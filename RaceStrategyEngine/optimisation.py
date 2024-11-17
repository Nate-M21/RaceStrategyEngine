"""This module defines the Optimization class, which is responsible for optimizing race strategies
based on tyre models.

The Optimization class provides functionality for:
- Handling various tyre models (linear, exponential, logarithmic, or piecewise functions)
- Optimizing race strategies to minimize total race time
- Calculating optimal tyre usage distributions for given strategies

This class serves as a foundation for the Combinatorics class to use to help select strategies and can be utilized
for specific live race scenarios when setting up used tyres."""

from collections import Counter
from dataclasses import asdict
from typing import Optional

import numpy as np
import sympy as sp
from scipy.optimize import OptimizeResult, minimize
from sympy import integrate, lambdify

from .tyre_model import TyreModel


class StrategyOptimization:
    """
    Class for the optimization of strategies based on the tyre models entered. Can optimise the strategy whether the
    tyre model is linear, exponential, logarithmic or a piecewise function

    Attributes:
        _tyre_models (dict): Information about tyre models.
        num_laps (int): Number of laps in the race.
        pit_loss (int): Time lost due to pitting.

    """

    def __init__(self, tyre_information: dict[str, TyreModel], num_laps: int, pit_loss: float):
        """Initializes the Optimization class with necessary race parameters and tyre information.

        Args:
            tyre_information (dict): Dictionary containing tyre models and their properties. 
            Values must be of type TyreModel.
            num_laps (int): Total number of laps in the race.
            pit_loss (int): Time lost per pit stop.

        Raises:
            TypeError: If any model in tyre_information is not an instance of TyreModel.
        """

        self._tyre_models = self._get_tyre_models(tyre_information)
        self.num_laps = num_laps
        self.pit_loss = pit_loss
        self._used_tyres = []
        self.lap_symbol = self.extract_tyre_model_symbol(next(iter(self._tyre_models.values()))['model'])

        # The strategy to be optimized
        self._strategy = []

        # The symbols that will be used for the optimization ie x, x2 etc
        self._symbols = []

    @property
    def num_laps(self) -> int:
        """The total number of laps in the race.

        Returns:
            int: The number of laps in the race.
        """
        return self._num_laps

    @num_laps.setter
    def num_laps(self, value) -> None:
        """Set the total number of laps for the race.

        Args:
            value (int): The number of laps to set for the race.

        Raises:
            ValueError: If the value is not a positive integer or is less than 2.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Number of laps must be a positive integer.")
        if value < 2:
            raise ValueError("Number of laps must be at least 2 to evaluate strategies and attempt to optimize them.")
        self._num_laps = value

    @property
    def pit_loss(self) -> int | float:
        """The time lost due to a pit stop.

        Returns:
            float: The time lost in seconds for each pit stop.
        """
        return self._pit_loss

    @pit_loss.setter
    def pit_loss(self, value) -> None:
        """Set the time lost due to a pit stop.

        Args:
            value (float): The time in seconds lost for each pit stop.

        Raises:
            ValueError: If the value is not a positive number.
        """
        self.validate_pit_loss(value)
        self._pit_loss = value
          
    @property
    def used_tyres(self) -> list[tuple[str, int]]:
        """List of tuples representing used tyres and their lap counts.

        Returns:
            list: A list of tuples, where each tuple contains (compound, laps).
        """
        return self._used_tyres

    @used_tyres.setter
    def used_tyres(self, value: list[tuple[str, int]]) -> None:
        """Sets the used tyres list after validating the input.

        Args:
            value (list): A list of tuples, where each tuple contains (compound, laps).

        Raises:
            TypeError: If the input is not a list.
            ValueError: If any tuple in the list is invalid or contains incorrect data.
        """
        # Zero laps on used tyres isn't logically
        self._validate_tyre_data(value, allow_zero_laps=False, context='used tyres')
        self._used_tyres = value

    @staticmethod
    def _get_tyre_models(tyre_information: dict[str, TyreModel]) -> dict:
        """Transform tyre information to a standardized dictionary format.

        This method ensures that all tyre models are instances of TyreModel and converts them to a consistent
        dictionary format for internal use.

        Args:
            tyre_information (dict): Dictionary of tyre models.

        Returns:
            dict: Standardized dictionary of tyre models.

        Raises:
            TypeError: If any model is not an instance of TyreModel.
        """
        if not all(isinstance(model, TyreModel) for model in tyre_information.values()):
            raise TypeError("All tyre models must be of data type / instance TyreModel")
        set_of_symbols = {StrategyOptimization.extract_tyre_model_symbol(tyre.model) for tyre in tyre_information.values()}
        if len(set_of_symbols) != 1:
            raise ValueError(f'All tyre models must use the same variable / symbol. Pick just ONE common symbol for ALL'
                             f' tyre models to use from the following entered: {set_of_symbols}')
        return {compound: asdict(model) for compound, model in tyre_information.items()}

    @staticmethod
    def extract_tyre_model_symbol(model: sp.Expr) -> sp.Symbol:
        """Extract any mathematical symbol / variable from a tyre model.

        This method identifies and returns the symbolic variable used in the tyre model expression,
        which typically represents the lap number.

        Args:
            model (sympy.Expr): The tyre model expression.

        Returns:
            sympy.Symbol: The common symbol used in the model.

        Raises:
            ValueError: If no symbols are found in the model.
            TypeError: If the extracted symbol is not an instance of sympy.Symbol.
        """
        if not model.free_symbols:
            raise ValueError("No symbols found in the model")
        symbol = next(iter(model.free_symbols))
        if not isinstance(symbol, sp.Symbol):
            raise TypeError(f"Expected type Symbol from sympy, got {type(symbol).__name__} instead")
        return symbol

    @staticmethod
    def _generate_symbols(base_symbol: sp.Symbol, number_of_compounds: int) -> list:
        """Generate symbols for multi-stint strategies.

        This method creates a list of symbolic variables based on the base symbol, which are used
        to represent different stints in a multi-stint race strategy. It generates one less symbol
        than the number of compounds, as the last compound's laps are implicitly determined.

        Args:
            base_symbol (sympy.Symbol): The base symbol to use.
            number_of_compounds (int): Total number of compounds in the strategy.

        Returns:
            list: List of sympy symbols, from x1 to x(number_of_compounds-1).
        """
        base_symbol_str = str(base_symbol)
        return [sp.symbols(f'{base_symbol_str}{i}') for i in range(1, number_of_compounds)]

    def _optimize_strategy(self, strategy: list[str]) -> OptimizeResult:
        """Optimize the race strategy for given tyre compounds.

        This method sets up the optimization problem for a given list of tyre compounds and performs
        the optimization to find the best distribution of laps for each compound.

        Args:
            strategy (list): List of tyre compounds to be used in the race strategy.

        Returns:
            scipy.optimize.OptimizeResult: The result of the optimization process.

        Raises:
            ValueError: If the strategy contains less than two compounds, includes unrecognized compounds,
                        or doesn't include compounds from used_tyres.
        """
        self._check_strategy_validity(strategy)

        self._strategy = strategy
        self._symbols = self._generate_symbols(self.lap_symbol, len(strategy))

        return self._perform_optimization()

    @staticmethod
    def strategy_comparison(required_strategy: list[str] | tuple[str] | None, proposed_strategy: list[str] | tuple[str],
                            available_strategy_constraint: Optional[list[str] | tuple[str]] = None,
                            show_extras=False,
                            order_output=True):
        """Compare a proposed strategy against a required strategy and available tyre constraints.

        This method checks if a proposed strategy satisfies the requirements of a given
        strategy and/or doesn't exceed available tyre constraints. It can be used to validate
        if a strategy includes certain required compounds or to compare two strategies.
        The method can also return information about extra compounds in the proposed strategy
        or provide an ordered output.

        Args:
            required_strategy (list or tuple or None): The strategy that must be included.
            If None, only available_strategy_constraint will be checked.
            proposed_strategy (list or tuple): The strategy to be compared.
            available_strategy_constraint (list or tuple, optional): The available tyre compounds.
                If provided, checks if the proposed strategy doesn't exceed available tyres. Defaults to None.
            show_extras (bool, optional): If True, return extra compounds in proposed strategy. Defaults to False.
            order_output (bool, optional): If True, return an ordered strategy that mimics the required_strategy.
                Defaults to True.

        Returns:
            tuple: (is_valid, result), where is_valid is a boolean indicating if the proposed
                strategy satisfies the requirements and availability constraints, and result depends on the input flags:
                - If show_extras is True, result is a dict of extra compounds.
                - If order_output is True, result is the ordered strategy according to the required_strategy.
                - Otherwise, result is the proposed strategy.

        Raises:
            ValueError: If both required_strategy and available_strategy_constraint are None,
            as this would result in a redundant check.
        """
        if show_extras and order_output:
            raise ValueError('Only one of show_extras or order_output can be True at a time.')
        if required_strategy is None and available_strategy_constraint is None:
            raise ValueError('Either required_strategy or available_strategy_constraint must be provided. '
                             'Otherwise it is a redundant check')

        if required_strategy is None:
            required_strategy = []
        required_compounds = Counter(required_strategy)
        proposed_compounds = Counter(proposed_strategy)


        # Check if proposed strategy has at least as many of each compound
        is_valid = all(proposed_compounds[compound] >= count for compound, count in required_compounds.items())

        # If provided check the proposed strategy can actually be performed by checking
        # it does not exceed the amounts specified in the available strategies
        if available_strategy_constraint:
            available_compounds = Counter(available_strategy_constraint)

            is_valid = all(proposed_compounds[compound] <= available_compounds[compound] for compound in proposed_compounds)

        # Calculate any extra compounds in the proposed strategy
        extras = proposed_compounds - required_compounds

        if show_extras and is_valid:
            return is_valid, dict(extras)

        if order_output and is_valid:
            # Start with the required strategy and extend with extras
            ordered_strategy = list(required_strategy)
            ordered_strategy.extend(extras.elements())
            return is_valid, ordered_strategy

        return is_valid, proposed_strategy

    def optimal_lap_distribution(self, compounds: list[str], used_tyres: Optional[list[tuple[str, int]]] = None )\
            -> list[tuple[str, float]] | None:
        """Returns the optimal distribution of laps for the given compounds.

        This method optimizes the race strategy for the given list of tyre compounds and returns
        the optimal distribution of laps for each compound.

        Args:
            compounds (list): List of tyre compounds to be used in the race strategy.

        Returns:
            list: A list of tuples, where each tuple contains (compound, optimal_laps).

        Note: 
            Prints a failure message if the optimization is unsuccessful.
        """
        # print('Used tyres entered', used_tyres)
        if used_tyres is not None:
            self.used_tyres = used_tyres
        # print('used tyres being used in optimizing', self.used_tyres)
        result = self._optimize_strategy(compounds)
        if result.success:
            # print(result.message)
            return list(zip(self._strategy, result.x))
        else:
            print(f'{result.message} - Optimization FAILED!: Failed Result - {list(zip(self._strategy, result.x))}')

    def _integrate_tyre_models(self) -> list[sp.Expr,]:
        """Integration of all the tyre models.

        This method performs the crucial task of integrating the tyre degradation models for each
        compound in the current strategy. It captures the cumulative effect of tyre wear over a stint,
        which is essential for accurately calculating the total time impact of using each tyre compound
        for a specific number of laps.

        Returns:
            list: List of integrated tyre model expressions for each compound in the strategy.
        """
        integrals = []

        for index, compound in enumerate(self._strategy):
            # Check if it is the last compound if it is use the formula below
            if index == len(self._strategy) - 1:
                integrals.append(
                    integrate(self._tyre_models[compound]['model']).subs(self.lap_symbol,
                                                                         self.num_laps - sum(self._symbols)))
            else:
                integrals.append(
                    integrate(self._tyre_models[compound]['model']).subs(self.lap_symbol, self._symbols[index]))

        return integrals

    def _perform_optimization(self) -> OptimizeResult:
        """Performs the optimization to minimize the total race time.

        This method sets up and executes the optimization problem to find the optimal distribution
        of laps for each tyre compound in the strategy. It uses the integrated tyre models and
        considers constraints such as total race laps and used tyre stints (if applicable).

        Returns:
            scipy.optimize.OptimizeResult: The result of the optimization process, containing
            the optimal lap distribution and other optimization details.
        """
        tyre_model_integrals = self._integrate_tyre_models()
        combination = sum(tyre_model_integrals) + self.pit_loss * (len(self._strategy) - 1)
        f_sym = lambdify(self._symbols, combination)

        def race_model(x):

            return f_sym(*x[:-1])
        # Reason for " f_sym(*x[:-1]) " ignoring the last value in x: The duration for the last tyre compound is
        # automatically calculated from the difference to the sum of the preceding values in the model.
        # It is derived as the remaining laps after assigning laps to other compounds.
        # See the _integrate_tyre_models method for how these sums influence the integration results.

        # Constraint to ensure the total laps do not exceed available laps
        constraints = [
            {'type': 'eq', 'fun': lambda x: self.num_laps - sum(x)}
        ]

        if self.used_tyres:
            # Iterate through the used tyres, keeping track of index and tyre data
            for index, (compound, laps_run) in enumerate(self.used_tyres):
                # Add a new constraint for each used tyre stint
                constraints.append({
                    # I am specifying that this is an equality constraint
                    'type': 'eq',

                    # Define the constraint function:
                    # x is the array of lap counts for each stint that the optimizer is considering
                    # I is the index of the current stint (captured from the enumerate loop)
                    # target is the number of laps actually used in this stint (same as 'laps')
                    # The function x[i] - target will equal 0 when x[i] equals target
                    'fun': lambda x, i=index, target_laps=laps_run: x[i] - target_laps,

                    # Note: The following is just a human-readable description of this constraint process.
                    # For the i th stint, make sure the number of laps I'm suggesting (x[i]) is exactly equal
                    # to the number of laps I've actually run (target_laps).

                    # For debugging and logging when I need to check, This doesn't affect the optimization,
                    'description': f'Stint {index + 1} ({compound}) must be exactly {laps_run} laps'
                })

        # boundaries to ensure that each compound does not exceed its limit and that is it
        # NEVER less than 1 for a compound to be run
        boundaries = [(1, self._tyre_models[compound]['max_laps']) for compound in self._strategy]
        initial_guess = np.ones(len(self._strategy)) * (self.num_laps / len(self._strategy))

        return minimize(race_model, initial_guess, bounds=boundaries, constraints=constraints, method='SLSQP')

    def _check_strategy_validity(self, strategy: list[str]):
        """Validate the proposed race strategy.

        This method checks if the proposed strategy is valid by ensuring it contains at least
        two compounds, includes all compounds from used tyres (if any), and only contains
        recognized tyre compounds.

        Args:
            strategy (list): List of tyre compounds to be used in the race strategy.

        Raises:
            ValueError: If the strategy is invalid due to any of the following reasons:
                - Contains less than two compounds
                - Doesn't include all compounds from used tyres
                - Contains unrecognized tyre compounds
        """

        if len(strategy) == 1:
            raise ValueError('Optimization requires at least two compounds to consider pit stops.')
        num_stops = len(strategy) - 1

        self.validate_num_laps_and_stops(self.num_laps, num_stops)

        if self.used_tyres:
            # Taking only the compounds not the laps run on them
            used_tyre_compounds = [compound for compound, _ in self.used_tyres]
            is_valid, _ = self.strategy_comparison(used_tyre_compounds, strategy)
            if not is_valid:
                raise ValueError(
                    f"Invalid strategy: {strategy}\n"
                    f"At minimum the proposed strategy must include all compounds from "
                    f"used_tyres: {used_tyre_compounds}\n"
                    f"To clear used tyres, call 'used_tyres.clear()' before optimization."
                )

        missing = [compound for compound in strategy if compound not in self._tyre_models]
        if missing:
            raise ValueError(
                f"Tyre compound(s) {', '.join(missing)} not recognized. "
                f"Available: {list(self._tyre_models.keys())}"
            )

    def _validate_tyre_data(self, value: list[tuple[str, int]], allow_zero_laps: bool = False,
                            context: str | None = None) -> None:
        """Validates the tyre data format.

        Args:
            value (list): A list of tuples, where each tuple contains (compound, laps).
            allow_zero_laps (bool): If True, allows laps to be 0. Default is False.

        Raises:
            TypeError: If the input is not a list.
            ValueError: If any tuple in the list is invalid or contains incorrect data.
        """
        message = f'- for the {context}'
        if not isinstance(value, list):
            raise TypeError(f"Tyre data must be a list of tuples {message if context else ''}")

        for tyre_set in value:
            if not (isinstance(tyre_set, (tuple, list)) and len(tyre_set) == 2):
                raise ValueError(
                    f"Each tyre set must be a tuple or list of (compound, laps) {message if context else ''}\n"
                    f" Error: {tyre_set}")

            compound, laps = tyre_set
            self.check_compound_validity(compound)

            if not isinstance(laps, int):
                raise ValueError(f"Invalid laps for compound {compound}: must be an integer "
                                 f"{message if context else ''}")

            if allow_zero_laps:
                if laps < 0:
                    raise ValueError(f"Invalid laps for compound {compound}: must be a non-negative integer"
                                     f" {message if context else ''}")
            else:
                if laps <= 0:
                    raise ValueError(f"Invalid laps for compound {compound}: must be a positive integer "
                                     f"{message if context else ''}")

    def check_compound_validity(self, compound: str) -> str:
        """Validate a tyre compound against the current tyre models.

        This is a reusable helper method used throughout the class to ensure that
        a given tyre compound is valid according to the current set of tyre models.

        Args:
            compound (str): The tyre compound to validate.

        Returns:
            str: The input compound, if it's valid.

        Raises:
            ValueError: If the compound is not found in the current tyre models.
                The error message includes a list of available compounds.

    """
        if compound not in self._tyre_models:
            raise ValueError(f"Tyre compound: {compound} not recognized. Available: {list(self._tyre_models.keys())}")

        return compound

    @staticmethod
    def validate_num_laps_and_stops(number_of_laps: int, number_of_stops: int) -> None:
        """Validates that the number of laps is sufficient for the given number of stops.

        This method checks if there are enough laps in the race to accommodate the
        specified number of pit stops. It ensures that there is at least one lap for
        each compound in the strategy.

        Args:
            number_of_laps (int): The total number of laps in the race.
            number_of_stops (int): The number of pit stops planned for the strategy.

        Raises:
            ValueError: If the number of laps is insufficient for the given number of stops.
                The error message includes the minimum required number of laps.
        """
        if number_of_laps < number_of_stops + 1:
            raise ValueError(f"Number of laps ({number_of_laps}) must be at least {number_of_stops + 1} "
                             f"to attempt to optimize a {number_of_stops}-stop strategy.")

    @staticmethod   
    def validate_pit_loss(pit_loss) -> None:
        """Validates the pit loss value.

        This method checks if the provided pit loss value is a positive number
        (integer or float). 

        Args:
            pit_loss (int or float): The time lost due to a pit stop.

        Raises:
            ValueError: If pit_loss is not a positive number.

        Returns:
            None       
        """
        if not isinstance(pit_loss, (int, float)) or pit_loss <= 0 or pit_loss == float('inf'):
            raise ValueError("Pit loss must be a positive number.")





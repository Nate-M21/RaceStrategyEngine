"""This module defines the Driver class, which optimises race strategies for drivers based on various tyre models (
clean air). Each driver is characterized by a set of tyre models, a consistency level, and a number of planned pit
stops. The Driver class utilizes these attributes to find the theoretical optimal race strategy in 'clean air' ie
racing alone through combinatorial analysis provided by the StrategyCombinatorics class. This is a precursor to being
tested in the RaceStrategyEngine to validate if it will actually be effective in a race scenario"""

from typing import Callable, Literal, Optional, Union
from collections import Counter

import pandas as pd
import re

from .combinatorics import StrategyCombinatorics
from .race_configuration import RaceConfiguration
from .tyre_model import TyreModel


class Driver:
    """Represents a driver and optimizes race strategies based on tyre models in 'clean air' conditions.

    This class encapsulates driver details and computes optimal race strategies using
    tyre models and pit loss. It leverages combinatorial analysis to determine
    the best theoretical strategies, which can later be validated in race simulations.

    Attributes:
        name (str): Driver's name.
        _race_config (RaceConfiguration): Race configuration parameters.
        _tyre_models (Dict[str, TyreModel]): Tyre models available to the driver,
            where keys are compound names and values are TyreModel instances.
        _num_of_stops (int): Planned number of pit stops during the race.
        lap_time_mean_deviation (float): Average deviation from ideal lap time in seconds.
        lap_time_std_dev (float): Standard deviation for lap time variability in seconds.
        min_lap_time_variation (float): Minimum variation applied to each lap time in seconds.
        mean_tyre_change_time (float): Average time taken to change tyres during a pit stop in seconds.
        std_dev_tyre_change_time (float): Standard deviation of tyre change time in seconds.
        strategy (StrategyCombinatorics): Instance for calculating race strategies.
        _selected_strategy (list): Best overall clean-air strategy determined.
        _lambdified_tyre_models (dict): Lambdified expressions of tyre models for quick evaluation.
        _alternate_strategies (list): Collection of top alternative strategies.
        available_tyres_method (str): Method to select available tyres

    The class uses the StrategyCombinatorics class to determine optimal strategies in 'clean air' conditions,
    serving as a precursor to race simulation testing for the actual strategy effectiveness.
    """

    def __init__(self, driver_name: str, tyre_models: dict[str, TyreModel], driver_consistency: dict,
                 number_of_stops: int,
                 race_configuration: RaceConfiguration, top_n_strategies_parameter: int = 1,
                 stops_depth_limit: Optional[int] = None,
                 alternate_strategies_method: Literal['breadth', 'depth', 'all'] = 'breadth',
                 start_compound: Optional[str] = None,
                 available_tyres_constraint: Optional[list[tuple[str, int]]] = None,
                 available_tyres_method: Literal['least_used', 'as_provided'] = 'least_used',
                 ) -> None:
        """
        Initialize a Driver instance with race strategy parameters.

        Args:
            driver_name (str): Name of the driver.
            tyre_models (dict): Tyre models available to the driver.
            driver_consistency (dict): Measure of driver's consistency metrics.
                Expected keys:
                - 'mean': Average deviation from ideal lap time (in seconds)
                - 'std_dev': Standard deviation for lap time variability (in seconds)
                - 'min_lap_time_variation': Minimum variation applied to each lap time (in seconds)
                - 'mean_tyre_change_time': Average time taken to change tyres during a pit stop (in seconds)
                - 'std_dev_tyre_change_time': Standard deviation of tyre change time (in seconds)
            number_of_stops (int): Planned number of pit stops.
            race_configuration (RaceConfiguration): Race configuration parameters.
            top_n_strategies_parameter (int, optional): Number of top strategies to consider. Defaults to 1.
            stops_depth_limit (int): Maximum pit stops to evaluate. Defaults to None.
            alternate_strategies_method (str): 
                Method for alternative strategies. Must be one of 'breadth', 'depth', or 'all'. 
                Defaults to 'breadth'.
            start_compound (str or None): Preferred starting tyre compound. Defaults to None.
            available_tyres_constraint (list, optional): The available tyres to use. 
            Defaults to None meaning no constraint
            available_tyres_method (str, optional): Method to select tyres.
            'least_used': Prefer tyres with fewer laps (default).
            'as_provided': Use tyres in the order they are provided.

        Raises:
            TypeError: If race_config is not an instance of RaceConfig.
        """

        # I am ensuring that a race config is used for a driver prevents inconsistencies that could come from
        # different drivers having different race parameters.
        if not isinstance(race_configuration, RaceConfiguration):
            raise TypeError("race_configuration must be an instance of RaceConfig")

        self.name = driver_name
        self._race_config = race_configuration
        self._tyre_models = tyre_models
        self._num_of_stops = number_of_stops

        # Default values represent characteristics of an 'average' driver.
        # These are used if specific data is not provided in the driver_consistency dictionary,
        # ensuring the simulation can run with reasonable values even with incomplete data.
        self.lap_time_mean_deviation = driver_consistency.get('mean', 0.25)
        self.lap_time_std_dev = driver_consistency.get('std_dev', 0.15)
        self.min_lap_time_variation = driver_consistency.get('min_lap_time_variation', 0.05)
        self.mean_tyre_change_time = driver_consistency.get('mean_tyre_change_time', 2.5)
        self.std_dev_tyre_change_time = driver_consistency.get('std_dev_tyre_change_time', 0.3)

        self.strategy = StrategyCombinatorics(self.tyre_models, self._num_of_stops,
                                              self._race_config.num_laps,
                                              self._race_config.pit_lane_time_loss,
                                              start_compound=start_compound,
                                              available_tyres_constraint=available_tyres_constraint
                                              )

        # This is the strategy the driver will start with
        self.selected_strategy = self.strategy.best_strategy()
        self._lambdified_tyre_models = self.strategy.get_lambda_tyre_models_expr()
        self.available_tyres_method = available_tyres_method

        # Defaults to the number of stops if stops_depth_limit is provided it will be set to that
        # when the whole list of alternate strategies are calculated
        self._highest_amount_of_stops_in_alternate_strategies = number_of_stops

        # Calculate alternate strategies directly
        # Once set don't change these
        self.__top_n_strategies_parameter = top_n_strategies_parameter
        self.__stops_depth_limit = stops_depth_limit
        self.__alternate_strategies_method = alternate_strategies_method

        self._alternate_strategies = self._get_top_n_fastest_strategies_per_stop(
            top_n=self.__top_n_strategies_parameter,
            max_stops=self.__stops_depth_limit,
            method=self.__alternate_strategies_method
        )

    @property
    def available_tyres_constraint(self) -> list[tuple[str, int]] | None:
        """Get the constraint on available tyres for the driver's race strategy.

        This property returns the constraint on available tyres, which is a list of tuples
        where each tuple contains a tyre compound and the number of laps already run on that compound.

        Returns:
            list or None: A list of (compound, laps) tuples representing the
            available tyres and their usage, or None if no constraint was set.
        """

        return self.strategy.available_tyres_constraint

    @property
    def available_compounds(self) -> list[str] | None:
        """Get the list of available tyre compounds for the driver.

        This property returns a list of tyre compounds that are available for use in
        the driver's race strategy. It is derived from the available tyres constraint.

        Returns:
            list or None: A list of tyre compound names (e.g., ['soft', 'medium', 'hard']),
            or None if no available tyres constraint was set.
        """

        return self.strategy.available_compounds

    @property
    def laps_used(self) -> list[int] | None:
        """Get the number of laps used for each available tyre compound.

        This property returns a list of integers representing the number of laps
        already run on each available tyre compound. The order corresponds to the
        order of compounds in the available_compounds property.

        Returns:
            list or None: A list of integers representing the number of laps
            used for each available compound, or None if no available tyres constraint was set.
        """
        return self.strategy.laps_used

    @property
    def strategy_options(self) -> pd.DataFrame:
        """
        Get the DataFrame containing all valid strategies and their performance metrics.

        This property provides access to the underlying strategy DataFrame, which contains
        vital information about each possible strategy, such as race time, compounds used,
        and optimal lap distribution.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a unique strategy,
            including columns for 'strategy', 'optimal_dist', 'racetime', and 'stops'.
        """
        return self.strategy.strategy_df

    @property
    def selected_strategy(self):
        """
        The currently selected race strategy.

        Returns:
            tuple: A tuple containing:
                - float: The estimated total race time for this strategy.
                - List[Tuple[str, float]]: A list of tuples, each representing a stint:
                    - str: The tyre compound for the stint.
                    - float: The number of laps for the stint.

        Example:
            (3600.5, ('soft', 20.5), ('medium', 34.5))
        """
        return self._selected_strategy

    @selected_strategy.setter
    def selected_strategy(self, new_strategy) -> None:
        """
        Set a new race strategy.

        This setter validates and sets a new race strategy. It ensures that the new strategy
        meets all required criteria before assigning it. This setter performs extensive 
        validation to ensure the integrity of the race strategy.

        Args:
            new_strategy tuple: 
                The new strategy to set. It should be in the format:
                (race_time, (compound1, laps1), (compound2, laps2), ...) or
                [race_time, (compound1, laps1), (compound2, laps2), ...]
                where race_time is a float, and each compound-laps pair is a tuple.

        Raises:
            ValueError: If the new strategy doesn't meet the required criteria:
                - Must have a positive race time.
                - Must have at least two different tyre compounds.
                - Total laps must equal the race distance.
                - Each stint must have a valid compound and positive number of laps.


        """

        self._check_strategy_feasibility(new_strategy)

        self._selected_strategy = new_strategy

    def _check_strategy_feasibility(self, new_strategy):
        """Check the feasibility of a new race strategy.

        This method validates the race strategy structure and checks if it can be
        performed with the available compounds.

        Args:
            new_strategy (tuple): The strategy to check.

        Raises:
            ValueError: If the strategy is not valid or cannot be performed with available compounds.
        """

        self._validate_race_strategy(new_strategy)
        if self.available_compounds:
            # Checking if the strategy can be performed with the compounds available
            _, *compound_strategy = new_strategy
            race_strategy_compounds, _ = zip(*compound_strategy)
            self._check_compounds_availability(strategy_compounds=race_strategy_compounds)

    def _check_compounds_availability(self, strategy_compounds):
        """Check if the strategy compounds are available.

        Args:
            strategy_compounds (tuple): The compounds used in the new strategy.

        Raises:
            ValueError: If the strategy cannot be performed with the available compounds.
        """
        is_valid, _ = self.strategy.strategy_comparison(None,
                                                        strategy_compounds,
                                                        self.available_compounds)
        if not is_valid:
            available_compounds = Counter(self.available_compounds)
            compound_count = ', '.join(f'{count} {compound}' for compound, count in available_compounds.items())
            raise ValueError(f'The strategy {strategy_compounds} for {self.name} '
                             f'is not possible based on your available tyres: '
                             f'{compound_count}')

    def strategy_creator(self, new_lap_distributions, optional_message: Optional[str] = None):
        """Create a new race strategy.

        This method creates a new strategy, validates it, and returns it in the correct format.

        Args:
            new_lap_distributions (list or tuple): The new strategy to create.
            optional_message (str, optional): An optional message to append to the strategy name.

        Returns:
            tuple: The created and validated strategy.

        Raises:
            ValueError: If the strategy format is invalid or if it fails validation.
        """

        new_lap_distributions = tuple([tuple(lap_distribution) for lap_distribution in new_lap_distributions])
        new_lap_distributions = tuple(new_lap_distributions)

        if all(isinstance(item, (tuple, list)) for item in new_lap_distributions):
            base_message = "User generated strategy (No theoretical clean air race time)"
            if optional_message:
                strategy_name = f"{base_message} + {optional_message}"
            else:
                strategy_name = base_message

            full_strategy = (strategy_name,) + new_lap_distributions

            self._check_strategy_feasibility(full_strategy)

            return full_strategy

        else:
            raise ValueError('Invalid strategy format')

    def get_unique_strategy_permutations(self, use_alternate_strategies: bool = False) \
            -> tuple[int, list[tuple[str, ...]]]:
        """ Get unique permutations of strategy combinations.

        This method generates permutations either from all strategies in the strategy
        DataFrame or from a subset of alternate strategies. Retrieves only the unique
        permutations of strategy combinations the driver is capable of doing or the unique permutations
        of the strategies that will be tested in Monte Carlo simulations as per the arguments
        for the alternate strategies method and parameters.

         Args:
            use_alternate_strategies (bool): If True, use only the alternate strategies
                selected for Monte Carlo simulations. If False, use all strategies
                from the strategy DataFrame.

        Returns:
            tuple: A tuple containing:
                - int: The number of unique permutations.
                - list: A list of unique compound strategy permutations.

        Notes:
            Does not take into account laps used on compounds. For example, a new soft
            and old soft are considered the same compound, so they won't be distinct
            in the permutations.
        """
        strategies_compounds = []

        if use_alternate_strategies:
            strategies = self.get_alternate_strategies()

            # Removing the extra information, and focusing just on the strategy_compounds
            for strategy in strategies:
                _, *compound_strategy = strategy
                strategy_compounds, laps = zip(*compound_strategy)
                strategies_compounds.append(strategy_compounds)

        else:
            # This is already in just strategy_compounds format of having only compounds
            strategies_compounds = self.strategy_options['strategy']

        permutations = self.strategy.strategy_permutations(strategies_compounds)
        return len(permutations), permutations

    def get_alternate_strategies(self, dict_mode: bool = False):
        """
        Retrieve alternate strategies with the option to return as a dictionary or list.

        This method provides access to the precomputed alternate strategies.
        It can return the strategies either as a list (default) or as a dictionary,
        depending on the dict_mode parameter.

        Args:
            dict_mode (bool): If True, return strategies as a dictionary. 
                            If False, return as a list. Defaults to False.

        Returns:
            list or dict: Alternate strategies in the specified format.
                If dict_mode is False, returns a list of strategy tuples.
                If dict_mode is True, returns a dictionary where keys are stop counts
                and values are lists of strategy tuples.

        Note:
            The internal _alternate_strategies attribute is always maintained as a list.
            When dict_mode is True, the method recomputes the strategies in dictionary format.
        """
        if dict_mode:
            # Recompute and return strategies in dictionary mode.
            # Note: The actual _alternate_strategies attribute is always stored as a list
            # to maintain consistency. This approach allows for flexible output
            # while keeping a standardized internal representation.
            return self._get_top_n_fastest_strategies_per_stop(
                top_n=self.__top_n_strategies_parameter,
                max_stops=self.__stops_depth_limit,
                method=self.__alternate_strategies_method,
                dict_mode=True
            )
        return self._alternate_strategies

    @property
    def highest_amount_of_stops_in_alternate_strategies(self) -> int:
        """Get the amount of pit stops within strategy that has the most pit stops among the alternate options.

        This property looks at all the calculated alternate strategies and returns
        the number of pit stops in the strategy with the highest number of stops.

        Returns:
            int: The single highest number of pit stops found in the alternate strategies.
        """
        return self._highest_amount_of_stops_in_alternate_strategies

    @property
    def lambdified_tyre_models(self) -> dict[str, Callable]:
        """
        Get the lambdified expressions of tyre models for quick evaluation.

        Returns:
            dict: A dictionary where keys are tyre compound names and 
                                 values are lambdified functions representing tyre performance models.
        """
        return self._lambdified_tyre_models

    @property
    def tyre_models(self) -> dict[str, TyreModel]:
        """Get the tyre models for the driver.

        Returns:
            dict: The tyre models associated with this driver.
        """
        return self._tyre_models

    @tyre_models.setter
    def tyre_models(self, new_tyre_models: dict[str, TyreModel]) -> None:
        """Set new tyre models for the driver and update related strategies.

        Args:
            new_tyre_models (Dict): New tyre models to be set.
        """
        self.update_tyre_models(new_tyre_models)

    @property
    def num_of_stops(self) -> int:
        """Get the number of planned pit stops.

        Returns:
            int: The number of planned pit stops for the race.
        """
        return self._num_of_stops

    @property
    def race_config(self) -> RaceConfiguration:
        """RaceConfig: Get the race configuration parameters.

        Returns:
            RaceConfiguration: The race configuration object associated with this driver.
        """
        return self._race_config

    def _update_strategy_dependent_attributes(self, keep_current_strategy: bool = False) -> None:
        """
        Update all attributes that depend on the strategy.

        This method recalculates and updates the following attributes:
        - _selected_strategy: The best overall strategy determined by the StrategyCombinatorics instance.
        - _lambdified_tyre_models: The lambdified expressions of tyre wear models for quick evaluation.
        - alternate_strategies: A collection of top alternative strategies based on the initial parameters.

        Note:
            This method should be called whenever there's a change that could affect the optimal strategy,
            such as updating tyre models or culling a tyre compound.
        """
        if not keep_current_strategy:  # basically if there is an update to tyre models or culling of tyre,
            # the default is to always select the best
            self._selected_strategy = self.strategy.best_strategy()
        self._lambdified_tyre_models = self.strategy.get_lambda_tyre_models_expr()

        self._alternate_strategies = self._get_top_n_fastest_strategies_per_stop(
            top_n=self.__top_n_strategies_parameter,
            max_stops=self.__stops_depth_limit,
            method=self.__alternate_strategies_method)

    def update_tyre_usage(self, used_tyres: list[tuple[str, int]], current_tyre: str) -> None:
        """
        Update the driver's tyre usage history and current tyre.

        This method updates the strategy calculations based on the tyres used so far in the race
        and the current tyre compound in use.

        Args:
            used_tyres (list of tuples): List of (compound, laps) tuples representing tyre usage history.
            current_tyre (str): The compound currently in use.

        Example:
            driver.update_used_tyres([('soft', 20), ('medium', 25)], 'hard')
            This indicates the driver has used soft tyres for 20 laps, medium for 25 laps,
            and is currently on hard tyres.
        """
        self.strategy.update_used_tyres_and_current_tyre(used_tyres, current_tyre)
        self._update_strategy_dependent_attributes()

    def update_tyre_models(self, new_tyre_models: dict[str, TyreModel]) -> None:
        """
        Update driver's tyre models and recalculate all related strategies and data.

        Args:
            new_tyre_models (dict): New tyre models data.

        Updates the driver's tyre models, recalculates the strategy using the
        existing Combinatorics object, and updates all related attributes to
        ensure consistency across the entire Driver object.
        """
        self.strategy.update_tyre_models(new_tyre_models)
        self._tyre_models = new_tyre_models
        self._update_strategy_dependent_attributes()

    def cull_tyre(self, compound: str) -> None:
        """
        Remove a tyre compound from consideration in all strategies.

        This method removes the specified tyre compound from all strategies and
        updates all related attributes of the Driver instance.

        Args:
            compound (str): The tyre compound to remove from all strategies.

        Returns:
            None: The method modifies the Driver instance in place.

        Note: This method is mainly to be used in a live race scenario where you see 
        the tyre performance of a compound isn't what you expected, and you want to
        completely remove it from consideration and processing of future predictions. 
        This additionally decreases the computations needed to be done for future predictions.
        """
        # Call the cull_tyre method from Combinatorics instance so self.strategy doesn't have the tyre anymore
        self.strategy.cull_tyre(compound)

        # # Update the driver's tyre models by removing it
        if compound in self.tyre_models:
            del self.tyre_models[compound]

        self._update_strategy_dependent_attributes()

    def _validate_race_strategy(self, proposed_strategy: tuple):
        """Validates the proposed race strategy.

        Args:
            proposed_strategy (tuple): A tuple containing race time and stints.
                Format: (race_time, (compound1, laps1), (compound2, laps2), ...)

        Raises:
            ValueError: If the strategy doesn't meet the required criteria:
                - Must have a positive race time.
                - Must have at least two different tyre compounds.
                - Total laps must equal the race distance.
                - Each stint must have a valid compound and positive number of laps.

        Returns:
            None
        """
        race_time = proposed_strategy[0]
        if isinstance(race_time, (int, float)):
            if race_time <= 0:
                raise ValueError("Race time must be a positive number")
        elif isinstance(race_time, str):
            if not re.match(r"User generated strategy.*", race_time):
                raise ValueError("Race time string must start with 'User generated strategy'")
        else:
            raise ValueError(
                "Race time must be either a positive number or a string starting with 'User generated strategy'")

        stints = list(proposed_strategy[1:])

        if not all(isinstance(stint, tuple) and len(stint) == 2 for stint in stints):
            raise ValueError("Stints must be tuples of (compound, laps)")

        if not all(
                isinstance(compound, str) and isinstance(laps, (int, float)) and self.strategy.check_compound_validity(
                    compound) and laps > 0 for compound, laps in stints):
            raise ValueError(
                f"Each stint must have a string compound that is valid compound name and positive number of laps. {proposed_strategy} ")

        # Check for at least two different compounds
        if len(set(compound for compound, _ in stints)) < 2:
            raise ValueError(
                "Strategy must use at least two different tyre compounds")

        # Check if total laps match the race distance
        total_laps = sum(laps for _, laps in stints)
        # Allow small floating-point discrepancies
        if abs(total_laps - self._race_config.num_laps) > 0.1:
            raise ValueError(
                f"Total laps in strategy ({
                total_laps}) must equal the total in the race "
                f"({self._race_config.num_laps})")

    def _validate_stops_depth_limit(self, depth_limit: Optional[int] = None) -> int:
        """
        Validates the stops depth limit to ensure it is within an acceptable range.

        If depth_limit is None, it defaults to the number of stops to be considered.

        Args:
            depth_limit (int, optional): The maximum number of pit stops to evaluate.
                If None, defaults to self.num_of_stops.

        Returns:
            int: The validated stops depth limit.

        Raises:
            TypeError: If depth_limit is not None and not an integer.
            ValueError: If depth_limit is not within the range [1, self.num_of_stops].
        """
        if depth_limit is None:
            return self.num_of_stops

        if not isinstance(depth_limit, int):
            raise TypeError(f"stops_depth_limit must be an integer between 1 and {
            self.num_of_stops} (inclusive)")

        if 1 <= depth_limit <= self.num_of_stops:
            return depth_limit

        raise ValueError(
            f"The stops_depth_limit ({depth_limit}) must be between 1 and {
            self.num_of_stops} "
            f"(inclusive) for {self.name}")

    @staticmethod
    def _validate_alternate_strategies_method(method: Optional[str] = None) -> str:
        """Validate the input for alternate_strategies_method.

        Args:
            method (str, optional): The method to validate. If None, defaults to 'breadth'.

        Returns:
            str: The validated method.

        Raises:
            ValueError: If the method is not one of 'breadth', 'depth', or 'all'.
        """
        if method is None:
            return 'breadth'
        valid = ['breadth', 'depth', 'all']
        if method not in valid:
            raise ValueError(
                f'Invalid methodology for the calculation of alternate_strategies. Valid are: {valid}')

        return method

    @staticmethod
    def _validate_top_n_strategies(number: int) -> int:
        """
        Validates that the number is an integer and at least 1 for the top n strategies.

        Args:
            number (int): The number of top strategies to consider.

        Returns:
            int: The validated number of top strategies.

        Raises:
            TypeError: If the input is not an integer.
            ValueError: If the input is less than 1.
        """
        if not isinstance(number, int):
            raise TypeError(
                'Invalid input. The number must be an integer for the top n strategies.')

        if number < 1:
            raise ValueError(
                'Invalid input. The number must be at least 1 for the top n strategies.')

        return number

    def _validate_args_for_top_n_method(
            self,
            top_n: int = 1,
            dict_mode: bool = False,
            num_stops_limit: Optional[int] = None,
            method: Optional[Literal['breadth', 'depth', 'all']] = None
    ) -> tuple[int, bool, int, str]:
        """
        Validate the inputs for the get_top_n_fastest_strategies_per_stop method.

        Args:
            top_n (int): Number of top strategies to consider. Defaults to 1.
            dict_mode (bool): Whether to return results as a dictionary. Defaults to False.
            num_stops_limit (int, optional): Maximum number of stops to consider. Defaults to self.num_of_stops if None.
            method (str, optional): Strategy selection method. Defaults to 'breadth' if None.

        Returns:
            tuple: Validated values (top_n, dict_mode, stops_limit, method_type)

        Raises:
            TypeError: If inputs are of incorrect type.
            ValueError: If inputs are out of valid ranges.
        """
        validated_top_n = self._validate_top_n_strategies(top_n)
        validated_stops_limit = self._validate_stops_depth_limit(
            num_stops_limit)
        validated_method = self._validate_alternate_strategies_method(method)

        if not isinstance(dict_mode, bool):
            raise TypeError("dict_mode must be a boolean")

        return validated_top_n, dict_mode, validated_stops_limit, validated_method

    def _get_top_n_fastest_strategies_per_stop(
            self,
            top_n: int = 1,
            dict_mode: bool = False,
            max_stops: Optional[int] = None,
            method: Optional[Literal['breadth', 'depth', 'all']] = None
    ) -> Union[list[tuple], dict[str, list[tuple]]]:
        """
        Retrieve the top N fastest strategies for each number of pit stops considered.

        This method calculates and returns the fastest race strategies based on the specified parameters.
        It's primarily used for memoization in race simulations, providing a baseline for Monte Carlo simulations.

        Args:
            top_n (int): Number of top strategies to retrieve for each pit stop count. Defaults to 1.
            dict_mode (bool, optional): If True, return strategies in a dictionary format. If False, return as a list.
            Defaults to False.
            max_stops (int, optional): The maximum number of pit stops to consider.Defaults to self.num_of_stops if None
            method (str, optional): The method to use for strategy selection. Defaults to 'breadth'
                - 'breadth': Select top N strategies for each pit stop count.
                - 'depth': Select overall top N strategies across all pit stop counts.
                - 'all': Select all strategies up to _max_stops_among_all_the_drivers.

        Returns:
            list or dict: The top strategies per stop count, either as a list or a dictionary.
            Each strategy is represented as a tuple of (race_time, list of (compound, laps) tuples).
        """

        validated_params = self._validate_args_for_top_n_method(
            top_n, dict_mode, max_stops, method)
        top_n, dict_mode, stops_limit, strategy_selection_method = validated_params

        strategies_dict = {}
        strategies_list = []

        # Note!: Mutability use
        # Leveraging the mutability of lists and dictionaries, the
        # changes made within these methods will affect the original objects.
        # This approach is used for efficiency, avoiding unnecessary copying.
        match strategy_selection_method:
            case 'breadth':
                self.__breadth_method(top_n, stops_limit, dict_mode, strategies_dict, strategies_list)
            case 'depth':
                self.__depth_method(top_n, stops_limit, dict_mode, strategies_dict, strategies_list)
            case 'all':
                self.__all_method(stops_limit, dict_mode, strategies_dict, strategies_list)
            # This case is unlikely due to prior error checking,
            # but it's included for completeness.
            case _:
                raise ValueError(
                    'Invalid methodology for calculating alternate strategies')

        return strategies_dict if dict_mode else strategies_list

    def __breadth_method(self, top_n: int, stops_limit: int, use_dict_mode: bool, strategies_dict: dict,
                         strategies_list: list):
        """Calculates top N fastest strategies for each number of pit stops.

        Args:
            top_n (int): Number of top strategies to retrieve for each pit stop count.
            stops_limit (int): Maximum number of pit stops to consider.
            use_dict_mode (bool): If True, store results in dictionary format.
            strategies_dict (dict): Dictionary to store strategies if use_dict_mode is True.
            strategies_list (list): List to store strategies if use_dict_mode is False.

        Returns:
            None: Results are stored in strategies_dict or strategies_list.

        Notes:
            - Leveraging mutability so there is no need for copying of the original list or dict.
        """
        self._highest_amount_of_stops_in_alternate_strategies = stops_limit
        # find the top n fastest strategy per strategy type
        for stop in range(1, stops_limit + 1):
            # get only the n stops strategy rows
            filtered_df = self.strategy.filter_by_stop(stop)
            # get top n (fastest) of those rows
            top_n_fastest_strat_df = self.strategy.find_n_lowest_strategies(
                top_n, filtered_df)
            # get race time and optimal dist of these strategies
            formatted_df = top_n_fastest_strat_df.apply(
                self.strategy.get_racetime_and_optimal_dist, axis=1)
            # Caution: Mutability use This is the mutable list or dict, whatever I add here will be visible in the
            # original object in the _get_top_n_fastest_strategies_per_stop
            if not formatted_df.empty:  # check if it is NOT empty if so add it
                # Caution: Mutability use This is the mutable list or dict, whatever I add here will be visible in
                # the original object in the _get_top_n_fastest_strategies_per_stop
                self.__store_strategy_results(
                    formatted_df, stop, use_dict_mode, strategies_dict, strategies_list)

    def __depth_method(self, top_n: int, stops_limit: int, use_dict_mode: bool, strategies_dict: dict,
                       strategies_list: list):
        """Calculates overall top N fastest strategies across all pit stop counts.

        Args:
            top_n (int): Number of top strategies to retrieve.
            stops_limit (int): Maximum number of pit stops to consider.
            use_dict_mode (bool): If True, store results in dictionary format.
            strategies_dict (dict): Dictionary to store strategies if use_dict_mode is True.
            strategies_list (list): List to store strategies if use_dict_mode is False.

        Returns:
            None: Results are stored in strategies_dict or strategies_list.

        Notes:
            - Leveraging mutability so there is no need for copying of the original list or dict.
        """
        # find the top n fastest strategies from the whole dataframe
        top_n_fastest_strat_df = self.strategy.find_n_lowest_strategies(
            nfastest=top_n)
        self._highest_amount_of_stops_in_alternate_strategies = top_n_fastest_strat_df['stops'].max(
        )

        # now loop through them and filter them by stop
        for stop in range(1, stops_limit + 1):
            # filter and find the n stop in each iteration
            filtered_df = self.strategy.filter_by_stop(
                stop, top_n_fastest_strat_df)
            # format this output
            formatted_df = filtered_df.apply(
                self.strategy.get_racetime_and_optimal_dist, axis=1)
            # if the dataframe is empty because that number of stops is not in the top n do not add it
            if not formatted_df.empty:  # check if it is NOT empty if so add it
                # Caution: Mutability use This is the mutable list or dict, whatever I add here will be visible in
                # the original object in the _get_top_n_fastest_strategies_per_stop
                self.__store_strategy_results(
                    formatted_df, stop, use_dict_mode, strategies_dict, strategies_list)

    def __all_method(self, stops_limit: int, use_dict_mode: bool, strategies_dict: dict, strategies_list: list):
        """Calculates all strategies up to the specified stop limit.

        Args:
            stops_limit (int): Maximum number of pit stops to consider.
            use_dict_mode (bool): If True, store results in dictionary format.
            strategies_dict (dict): Dictionary to store strategies if use_dict_mode is True.
            strategies_list (list): List to store strategies if use_dict_mode is False.

        Returns:
            None: Results are stored in strategies_dict or strategies_list.

        Notes:
            - Leveraging mutability so there is no need for copying of the original list or dict.
        """
        # just get all the strategies unless a stops limit is given then get all strategies less than the limit
        self._highest_amount_of_stops_in_alternate_strategies = stops_limit

        for stop in range(1, stops_limit + 1):
            # filter and find the n stops in each iteration from the whole dataframe
            filtered_df = self.strategy.filter_by_stop(stop)

            formatted_df = filtered_df.apply(
                self.strategy.get_racetime_and_optimal_dist, axis=1)
            # Caution: Mutability use.
            # This is the mutable list or dict, whatever I add here will be visible in the
            # original object in the _get_top_n_fastest_strategies_per_stop
            if not formatted_df.empty:  # check if it is NOT empty if so add it
                # Caution: Mutability use This is the mutable list or dict, whatever I add here will be visible in
                # the original object in the _get_top_n_fastest_strategies_per_stop
                self.__store_strategy_results(
                    formatted_df, stop, use_dict_mode, strategies_dict, strategies_list)

    @staticmethod
    def __store_strategy_results(formatted_df: pd.Series, stop_count: int, use_dict_mode: bool,
                                 strategies_dict: dict, strategies_list: list):
        """Stores strategy results in either a dictionary or a list.

        Args:
            formatted_df (pandas.Series): Series containing strategy results.
            stop_count (int): Number of stops for the current strategy.
            use_dict_mode (bool): If True, store results in dictionary format.
            strategies_dict (dict): Dictionary to store strategies if use_dict_mode is True.
            strategies_list (list): List to store strategies if use_dict_mode is False.

        Returns:
            None: Results are stored in strategies_dict or strategies_list.

        Notes:
            - Leveraging mutability so there is no need for copying of the original list or dict.
        """
        result_list = list(formatted_df)
        if use_dict_mode:
            strategies_dict[f'{stop_count}_stop'] = result_list
        else:
            strategies_list.extend(result_list)

"""This module implements the StrategyCombinatorics class, which extends the Optimization class
to handle complex race strategy combination, filtering and selection.

The StrategyCombinatorics class provides functionality for:
- Generating and evaluating race strategies based on tyre combinations
- Creating and updating strategy dataframes
- Comparing and filtering strategies
- Visualizing strategy performance (clean-air) and tyre degradation
"""


from typing import Optional, Callable, Literal, Sequence
from itertools import combinations_with_replacement, permutations
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
from word2number import w2n

from .optimisation import StrategyOptimization
from .tyre_model import TyreModel


class StrategyCombinatorics(StrategyOptimization):
    """
    Class for generating and evaluating race strategies based on tyre combinations.
    Extends the StrategyOptimization class to handle complex strategy calculations and comparisons
    by performing combinatorial optimization.

    Attributes:
        _number_of_stops (int): Maximum number of pit stops to consider.
        _start_compound (str | None): Initial tyre compound for strategy calculation.
        _available_tyres_constraint (list[tuple[str, int]] | None): Constraint on available tyres.
        _available_compounds (list[str] | None): List of available tyre compounds.
        _laps_used (list[int] | None): Number of laps used for each available compound.
        _strategy_df (pd.DataFrame): DataFrame containing all valid strategies and their performance.
        num_laps (int): Total number of laps in the race.
        pit_loss (float): Time lost during a pit stop.
        _tyre_models (dict[str, TyreModel]): Information about available tyre compounds and their models.

    The class provides methods for:
    - Generating valid tyre combinations
    - Creating and updating strategy dataframes
    - Comparing and filtering strategies
    - Visualizing strategy performance and tyre degradation
    """

    def __init__(self, tyre_information: dict[str, TyreModel], number_of_stops: int, num_laps: int, pit_loss: float,
                 start_compound: Optional[str] = None,
                 available_tyres_constraint: Optional[list[tuple[str, int]]] = None):
        """Initialize the StrategyCombinatorics object.

        Initializes the strategy dataframe, which contains vital information
        about each possible strategy, such as race time and compounds used. If a start compound is
        specified, it filters the initial strategies accordingly.

        Args:
            tyre_information (dict[str, TyreModel]): Information about available tyre compounds.
            number_of_stops (int): Maximum number of pit stops to consider.
            num_laps (int): Total number of laps in the race.
            pit_loss (float): Time lost during a pit stop.
            start_compound (str | None, optional): Initial tyre compound for strategy calculation. Defaults to None.
            available_tyres_constraint (list[tuple[str, int]] | None, optional): The available tyres to use.
                Defaults to None meaning no constraint.

        Raises:
            TypeError: If start_compound is not a valid tyre compound.
        """
        super().__init__(tyre_information, num_laps, pit_loss)
        self._number_of_stops = number_of_stops  # The number of stops you would like to consider

        self.validate_num_laps_and_stops(self.num_laps, self._number_of_stops)

        # Every strategy that has this compound will start with it, additionally if a strategy doesn't have it,
        # it is 'pruned'
        self._start_compound = self._check_valid_starting_compound(start_compound)
        self._available_tyres_constraint = self._validate_available_tyres(available_tyres_constraint)
        if self._available_tyres_constraint:
            self._available_compounds, self._laps_used = zip(*self._available_tyres_constraint)
        else:
            self._available_compounds, self._laps_used = None, None

        # Creates a 'strategy' dataframe that has vital information about the strategies, such race time,
        # compounds used etc
        filter_strategy = [self._start_compound] if self._start_compound is not None else None
        # if the start compound is given I can knock off a few strategies
        self._strategy_df = self._create_strategies_dataframe(filter_strategy)
        # primary use for live data scenarios to avoid computing strategies that cant happen now

    @property
    def strategy_df(self) -> pd.DataFrame:
        """
        Get the strategy dataframe for instance that shows free-air optimization results.

        Returns:
            pd.Dataframe: A pandas dataframe with strategy results
        
        """
        return self._strategy_df

    @property
    def available_tyres_constraint(self) -> list[tuple[str, int]] | None:
        """Get the available tyres' constraint.

        This property returns the constraint on available tyres for the race strategy.
        The constraint is typically a list of tuples, where each tuple contains a tyre
        compound and the number of laps already run on that compound.

        Returns:
            list or None: A list of (compound, laps) tuples representing the
            available tyres and their usage, or None if no constraint was set.
        """

        return self._available_tyres_constraint

    @property
    def available_compounds(self) -> tuple[str] | None:
        """Get the available tyre compounds.

        This property returns a tuple of tyre compounds that are available for use in
        the race strategy. It is derived from the available tyres' constraint.

        Returns:
            tuple or None: A tuple of tyre compound names (e.g., ('soft', 'medium', 'hard')),
            or None if no available tyres constraint was set.
        """
        return self._available_compounds

    @property
    def laps_used(self) -> tuple[int] | None:
        """Get the number of laps used for each available tyre compound.

        This property returns a tuple of integers representing the number of laps
        already run on each available tyre compound. The order corresponds to the
        order of compounds in the available_compounds property.

        Returns:
            tuple or None: A tuple of integers representing the number of laps
            used for each available compound, or None if no available tyres constraint was set.
        """
        return self._laps_used

    def generate_combinations(self, filter_strategy_by: Optional[list] = None, minimum_stops: int = 1) -> list:
        """Generate all valid tyre compound combinations for the race.

        This method creates a list of all possible tyre compound combinations based on the
        number of stops and available compounds. It ensures that each strategy uses at least
        two different compounds and can complete the entire race distance. If a filter_strategy
        is provided, it only includes combinations that satisfy the filter conditions.

        Args:
            minimum_stops: The minimum amount of stops, default value is 1
                because in F1 Sporting regulations you must stop at least once
            filter_strategy_by (list or None): List of compounds to filter strategies by, or None.

        Returns:
            list: List of valid tyre compound combinations.

        Raises:
            TypeError: If filter_strategy is not a list or None.
            ValueError: If no valid combinations are found that satisfy all constraints.
    """
        if filter_strategy_by is not None and not isinstance(filter_strategy_by, (tuple, list)):
            raise TypeError(
                'The filter_strategy must be a list | tuple. If it\'s a single compound, enter it as a single-element list.')

        if filter_strategy_by is not None:
            for compound in filter_strategy_by:
                self.check_compound_validity(compound)

        valid_strategies = []
        min_len_of_strategy_compounds = minimum_stops + 1
        max_len_of_strategy_compounds = self._number_of_stops + 1

        strategies = []
        for i in range(min_len_of_strategy_compounds, max_len_of_strategy_compounds + 1):
            tyre_combination: tuple[str]  # Type hint not necessary
            for tyre_combination in combinations_with_replacement(self._tyre_models, i):
                # I am making sure that there is a change of compound, I can remove it
                # Convert tyre_combination to a list
                # tyre_combination = list(tyre_combination)
                if len(set(tyre_combination)) > 1:
                    # if it's different racing series
                    if filter_strategy_by is None and self._available_compounds is None:
                        strategies.append(tyre_combination)
                    else:
                        # Checking if tyre_combination has at minium the required compounds
                        # that the required strategy is set have. ie filter_strategy
                        is_valid, ordered_tyre_combination = self.strategy_comparison(filter_strategy_by,
                                                                                      tyre_combination,
                                                                                      self._available_compounds,
                                                                                      order_output=True)
                        if is_valid:
                            strategies.append(ordered_tyre_combination)

        # ensuring the specific combination of compounds in a strategy can complete a whole race whilst
        # adhering to the max laps the user has set
        for strategy in strategies:
            sum_of_laps = 0
            for compound in strategy:
                compound_tyre_life = self._tyre_models[compound]['max_laps']  # for readability
                sum_of_laps += compound_tyre_life
            if sum_of_laps > self.num_laps:
                valid_strategies.append(strategy)

        if not valid_strategies:
            raise ValueError(
                f'There are no valid combinations that can run the entire race distance of {self.num_laps} laps.\n '
                f'Whilst adhering to your limit of only {self._number_of_stops}-stop(s) to consider '
                f'and with the constraint of max amount of laps set to run on each compound')

        return valid_strategies

    def _create_strategies_dataframe(self, filter_strategy: Optional[list] = None):
        """Create a DataFrame containing all valid strategies and their performance.

        This method generates all valid tyre combinations, optimizes each strategy, and
        compiles the results into a DataFrame. Each row in the DataFrame represents a
        unique strategy and includes information such as the compounds used, optimal
        lap distribution, total race time, and number of stops.

        Args:
            filter_strategy (list or None, optional): List of compounds to filter strategies by. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing strategy information, including columns for
                        'strategy', 'optimal_dist', 'racetime', and 'stops'.
        """
        strategies = self.generate_combinations(filter_strategy)
        failed_strategies = []

        data = {'strategy': [], 'optimal_dist': [], 'racetime': [], 'stops': []}

        # Optimise each valid strategy and its information to the dictionary
        for strategy in strategies:

            result = self._optimize_strategy(strategy)
            if result.success:  # Only add successful optimizations
                data['strategy'].append(strategy)
                data['racetime'].append(result.fun)
                data['optimal_dist'].append(list(zip(strategy, result.x)))
                data['stops'].append(len(strategy) - 1)
            else:
                failed_strategies.append(strategy)

        if not data['strategy']:
            raise ValueError(f"All strategy optimizations failed. Failed strategies: {failed_strategies}")

        return pd.DataFrame(data)

    def update_used_tyres_and_current_tyre(self, used_tyres: list[tuple[str, int]], current_tyre: str) -> None:
        """
        Update information about used tyres and the current tyre, then recalculate strategies.

        This method updates the tyre usage information for a race, including multiple stints
        on the same compound if applicable. It also considers the current tyre in use and
        recalculates valid strategies based on this updated information.

        Args:
            used_tyres (list of tuples): A list where each tuple contains two elements:
                - compound (str): The tyre compound used
                - laps (int): The number of laps run on that compound
                Can be an empty list if no tyres have been used yet.
            current_tyre (str): The compound currently in use

        Example:
            update_used_tyres_and_current_tyre([('soft', 1), ('soft', 5)], 'medium')
            This indicates the driver has used soft tyres for two stints (1 lap and 5 laps)
            and is currently on medium tyres.

            update_used_tyres_and_current_tyre([], 'soft')
            This indicates the race is starting with soft tyres and no previous tyre usage.
        """
        # Validate the current tyre compound
        self.check_compound_validity(current_tyre)

        # Process and validate each used tyre stint
        for compound, _ in used_tyres:
            self.check_compound_validity(compound)

        # Update the used tyres information using the setter
        self.used_tyres = used_tyres

        # Create combination filter
        combination_filter = [compound for compound, _ in used_tyres] + [current_tyre]
        # combination filter should never be empty! If there aren't any used tyre there has to be a current tyre

        # Recalculate strategies based on the updated information
        self._strategy_df = self._create_strategies_dataframe(combination_filter)

    def cull_tyre(self, tyre_to_remove: str) -> None:
        """Remove all strategies from the DataFrame that include the specified tyre compound.

        This method filters out all strategies that use the specified tyre compound.
        It modifies the strategy_df in place, removing any rows that contain the
        tyre_to_remove in their strategy. It also removes the tyre from the tyre_models
        dictionary to ensure it's not considered in future calculations.

        Args:
            tyre_to_remove (str): The tyre compound to remove from all strategies.

        Raises:
            ValueError: If tyre_to_remove is not a valid compound in the current tyre models.
    """
        self.check_compound_validity(tyre_to_remove)
        # Filter out rows containing the tyre_to_remove
        self._strategy_df = self._strategy_df[~self._strategy_df['strategy'].apply(lambda x: tyre_to_remove in x)]

        # completely removing it from the tyre models,so when recalculations happens in the future it's not considered
        del self._tyre_models[tyre_to_remove]

        # Resetting the index, so it looks nice, no other reason but aesthetic
        self._strategy_df.reset_index(drop=True, inplace=True)

    def update_tyre_models(self, new_tyre_models: dict[str, TyreModel],
                           used_tyres: Optional[list[tuple[str, int]]] = None, current_tyre: Optional[str] = None):
        """Update the tyre models and recalculate the strategy DataFrame.

        This method updates the tyre models with new data and recalculates
        the strategy DataFrame based on these new models. If new used tyres
        or current tyre information is provided, it updates this information
        as well before recalculating strategies.

        Args:
            new_tyre_models (dict): A dictionary containing the new tyre models. 
                                    Each key should be a tyre compound name 
                                    (e.g., 'soft', 'medium', 'hard'), and each value 
                                    should be a TyreModel object.
            used_tyres (list of tuples, optional): A list where each tuple contains 
                                                two elements:
                - compound (str): The tyre compound used
                - laps (int): The number of laps run on that compound
                Defaults to an empty list, which means no change to current used tyres.
            current_tyre (str, optional): The compound currently in use. 
                                        If None, assumes no change to current tyre.

        Returns:
            None
        """

        if used_tyres is not None and current_tyre is None:
            raise ValueError("If used_tyres is provided, current_tyre must also be specified.")

        if used_tyres is None:
            used_tyres = []

        for tyre in new_tyre_models:
            self.check_compound_validity(tyre)

        # Updating the Optimization part of the object with new tyre models, to avoid creating a whole new one
        super().__init__(new_tyre_models, self.num_laps, self.pit_loss)

        # Update strategies based on provided information
        if not used_tyres and current_tyre is None:
            # If no new tyre usage information, just recalculate strategies
            # with the updated tyre models
            self._strategy_df = self._create_strategies_dataframe()
        else:
            # If new tyre usage information is provided, update it and then
            # recalculate strategies
            self.update_used_tyres_and_current_tyre(used_tyres, current_tyre)

    def best_strategy(self, dist_mode: bool = True):
        """Find the fastest strategy amongst all combinations.

        This method identifies the optimal strategy from all possible combinations
        considered. It can return either the full strategy information or just the
        optimal lap distribution, depending on the dist_mode parameter.

        Args:
            dist_mode (bool, optional): If True, return only the lap distribution. 
                                        If False, return full strategy information.
                                        Defaults to True.

        Returns:
            tuple or pd.DataFrame: If dist_mode is True, returns a tuple containing
                                the race time and optimal lap distribution. 
                                If False, returns a DataFrame row with full 
                                strategy information including compound sequence,
                                race time, and number of stops.
        """

        if not dist_mode:  # Show full strategy information
            return self._get_fastest_strategy_info(self._strategy_df)
        else:  # standard is to just show the distribution of laps
            row_of_fastest = self._find_lowest_time(self._strategy_df)  # find the lowest strategy in all strategies and
            return self.get_racetime_and_optimal_dist(row_of_fastest)  # return the optimal dist

    def __getattr__(self, name: str):
        """Handle dynamic attribute access for optimizing strategies.

        This method allows for dynamic querying of optimal strategies based on the
        number of stops. It interprets method names in the format 'best_X_stop'
        where X can be a number (e.g., 'best_2_stop') or a word (e.g., 'best_two_stop').
        It then calls the appropriate method to find the best strategy for that
        number of stops.

        Args:
            name (str): The name of the attribute being accessed, expected to be
                        in the format 'best_X_stop'.

        Returns:
            The result of _find_best_n_stop_strategy for the specified number of stops.

        Raises:
            AttributeError: If the attribute name does not match the expected pattern
                            or if the number of stops cannot be interpreted.
        """

        if name.startswith('best_') and name.endswith('_stop'):
            middle_part = name[5:-5].replace('_', ' ')
            try:
                # Attempt to convert directly to an integer
                num_stops = int(middle_part)
            except ValueError:
                try:
                    # Attempt to convert words to numbers
                    # Remove any hyphens for compound words (e.g., twenty-one)
                    num_stops = w2n.word_to_num(middle_part.replace('-', ' '))
                except ValueError:
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

            return self._find_best_n_stop_strategy(num_stops)

        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _find_best_n_stop_strategy(self, amount_of_stops: int):
        """Find the best strategy for a specific number of stops.

        This method filters the strategies to only those with the specified number
        of stops, then identifies the fastest among them. It returns the race time
        and optimal lap distribution for this best strategy.

        Args:
            amount_of_stops (int): The number of stops to optimize for.

        Returns:
            tuple: A tuple containing two elements:
                - float: The total race time for the best strategy.
                - list: The optimal lap distribution, where each element is a
                        tuple of (compound, number of laps).
        """

        filtered_strategy_df = self.filter_by_stop(amount_of_stops)
        if filtered_strategy_df.empty:
            return None
        # send the filtered dataframe with only selected amount of stops to a method that find the
        # optimal dist of the fastest strategy
        row_of_fastest = self._find_lowest_time(filtered_strategy_df)
        return self.get_racetime_and_optimal_dist(row_of_fastest)

    @staticmethod
    def _find_lowest_time(dataframe: pd.DataFrame):
        """Find the fastest strategy in the given DataFrame.

        This method identifies the strategy with the lowest total race time
        from the provided DataFrame. It's typically used as a helper method
        for other optimization functions.

        Args:
            dataframe (pd.DataFrame): DataFrame containing strategy information,
                                    must include a 'racetime' column.

        Returns:
            pd.Series: The row corresponding to the fastest strategy, containing
                    all strategy information such as compounds used, lap
                    distribution, and race time.
        """

        lowest_race_time = dataframe['racetime'].idxmin()  # finding the index of the fastest strategy in dataframe
        row = dataframe.loc[lowest_race_time]  # getting all the information from that index row

        return row

    def _get_fastest_strategy_info(self, dataframe: pd.DataFrame):
        """Get full information about the fastest strategy in the given DataFrame.

        This method identifies the fastest strategy in the DataFrame and returns
        a new DataFrame containing only that strategy's information. It's useful
        when detailed information about the best strategy is needed.

        Args:
            dataframe (pd.DataFrame): DataFrame containing strategy information,
                                    must include a 'racetime' column.

        Returns:
            pd.DataFrame: A single-row DataFrame with full information about the
                        fastest strategy, including compound sequence, lap
                        distribution, race time, and number of stops.
        """

        min_row = self._find_lowest_time(dataframe)

        return min_row.to_frame().T

    def find_n_lowest_strategies(self, nfastest: int = 3, optional_df: pd.DataFrame = None) -> pd.DataFrame:
        """Find the n fastest strategies from the given or default DataFrame.

        This method returns the n strategies with the lowest race times.

        Args:
            nfastest (int, optional): The number of fastest strategies to return. Defaults to 3.
            optional_df (pd.DataFrame, optional): A DataFrame to use instead of self._strategy_df.
                If None, self._strategy_df is used. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the n fastest strategies, sorted by race time.

        Note:
            The returned DataFrame is a subset of either optional_df or self._strategy_df,
            containing only the rows corresponding to the n fastest strategies.
        """
        # Choose the appropriate DataFrame based on the method's input
        df_to_use = optional_df if optional_df is not None else self._strategy_df

        # Find indices of the n smallest race times
        indices_of_fastest = df_to_use['racetime'].nsmallest(nfastest).index

        # Return the DataFrame rows corresponding to these indices
        return df_to_use.loc[indices_of_fastest]

    def _cumsum_of_avg_lap_time_of_fastest_strategy(self) -> np.ndarray:
        """Calculate cumulative sum of average lap times for the fastest strategy.

        This method finds the fastest strategy, calculates its average lap time,
        and then computes the cumulative sum of this average time for all laps.

        Returns:
            np.ndarray: An array of cumulative times, where each element represents
                the cumulative time up to that lap for the fastest strategy.

        Note:
            The average lap time is calculated by dividing the total race time
            of the fastest strategy by the number of laps. This average is then
            used for all laps to create the cumulative sum.
        """
        df = self.find_n_lowest_strategies(1)
        average_lap_time = df['racetime'] / self.num_laps

        #  df['racetime'] is a single value, im  converting it to a float:
        average_lap_time_value = float(average_lap_time.item()) if isinstance(average_lap_time, pd.Series) else float(
            average_lap_time)

        average_list = np.full(self.num_laps, average_lap_time_value)

        cumulative_time = np.cumsum(average_list)

        return cumulative_time

    @staticmethod
    def match_strategy_compounds_and_laps_used(strategy_compounds: list[str],
                                               available_tyres_constraint: list[tuple[str, int]],
                                               driver_name: Optional[str] = None,
                                               *,
                                               method: Literal['least_used', 'as_provided'] = 'least_used',
                                               ) -> list[int]:
        """Match strategy compounds with available tyres and return laps used for each compound.

        This method takes a proposed strategy and available tyres, then attempts to match
        each compound in the strategy with the least used tyre of that compound type.

        Args:
            strategy_compounds (list): List of tyre compounds in the proposed strategy.
            available_tyres_constraint (list): List of available tyres and their usage.
                Each tuple contains (compound_name, laps_used).
            driver_name (str, optional): Name of the driver for error reporting. Defaults to None.
            method (str, optional): Method to select tyres.
            'least_used': Prefer tyres with fewer laps (default).
            'as_provided': Use tyres in the order they are provided.

        Returns:
            list: List of laps used for each compound in the strategy.

        Raises:
            ValueError: If the strategy cannot be fulfilled with the available tyres.

        Example:
            strategy = ['soft', 'medium', 'hard']
            available = [('soft', 5), ('medium', 3), ('hard', 0), ('soft', 0), ('medium', 1) ]
            match_strategy_compounds_and_laps_used(strategy, available)
            [0, 1, 0]
            match_strategy_compounds_and_laps_used(strategy, available, method='as_provided')
            [5, 3, 0]
        """
        if method not in ['least_used', 'as_provided']:
            raise ValueError("Invalid method. Must be either 'least_used' or 'as_provided'.")
        # Group available compounds by type
        compound_groups = defaultdict(list)
        for compound, laps in available_tyres_constraint:
            compound_groups[compound].append(laps)

        # Sort each group by laps used (prefer less used tyres) defualt
        if method == 'least_used':
            for compound in compound_groups:
                compound_groups[compound].sort()

        result = []
        for selected_compound in strategy_compounds:
            if selected_compound in compound_groups and compound_groups[selected_compound]:
                # Use the compound with the least laps
                laps = compound_groups[selected_compound].pop(0)
                result.append((selected_compound, laps))

            else:
                # If a driver name is not provided I am defaulting to a generic statement
                driver_name = driver_name or 'your race strategy'
                compounds = [compound for compound, _ in available_tyres_constraint]
                available_compounds = Counter(compounds)
                compound_count = ', '.join(f'{count} {compound}' for compound, count in available_compounds.items())
                raise ValueError(f'The strategy {strategy_compounds} for {driver_name} '
                                 f'is not possible based on your available tyres: '
                                 f'{compound_count}')

        # Extract only the laps used from the result
        laps_used_for_compounds = [laps_used for _, laps_used in result]

        return laps_used_for_compounds

    @staticmethod
    def smart_round(numbers: list, target_sum: int, scale_by_contribution: Optional[bool] = False) -> list[int]:
        """Round list of numbers while maintaining their distribution and to a target sum.

        This method is crucial for transforming float/decimal numbers received 
        from the optimization process for pit laps into practical, integer-based lap counts. 
        The optimization (SQLSP), often produces non-integer optimal lap counts 
        (e.g., run soft compound for 19.4 laps, medium for 24.3 laps and hard for 31.3 laps). However, 
        in real-world racing, pit stops occur at whole lap numbers. 
        This method bridges the gap between theoretical optimization and 
        practical application, ensuring that:
        1. The sum of rounded numbers equals the target sum (total race laps).
        2. Each rounded number is at least 1 (no zero-lap stints).
        3. The original distribution of numbers is preserved as much as possible,
        maintaining the optimized strategy's intent.

        Args:
            numbers (list): The numbers to be rounded, representing optimized counts for each element.
            target_sum (int): The desired sum of the rounded numbers, typically the total count to be distributed.
            scale_by_contribution (bool, optional): If True, scales the input numbers by their relative
                contribution to the target sum before rounding. Defaults to False.


        Returns:
            np.ndarray: An array of rounded integers that sum to target_sum, representing practical stint lengths.

        Note:
            The method uses an iterative approach, incrementing or decrementing
            values one by one to reach the target sum while minimizing changes
            to the original distribution. This ensures the final strategy remains
            as close as possible to the mathematically optimal solution while being 
            practically implementable.
        """
        if scale_by_contribution:
            numbers = StrategyCombinatorics.scale_values(numbers, target_sum)
        numbers = np.array(numbers)
        rounded_numbers = np.round(numbers)

        # Ensure all values are initially at least 1
        rounded_numbers = np.maximum(rounded_numbers, 1)
        current_sum = np.sum(rounded_numbers)

        while current_sum != target_sum:
            if current_sum < target_sum:
                # Need to increase the sum, find the best candidate to round up
                differences = numbers - rounded_numbers
                # Only consider indices where rounded_numbers are already >= 1
                index_to_increment = np.argmin(differences)
                rounded_numbers[index_to_increment] += 1
            elif current_sum > target_sum:
                # Need to decrease the sum, find the best candidate to round down
                differences = rounded_numbers - numbers
                # Only consider indices where rounded_numbers are > 1
                valid_indices = [i for i, val in enumerate(rounded_numbers) if val > 1]
                if not valid_indices:
                    break  # Prevent infinite loop if no valid indices to decrement
                index_to_decrement = valid_indices[np.argmin(differences[valid_indices])]
                rounded_numbers[index_to_decrement] -= 1

            current_sum = np.sum(rounded_numbers)

        return [int(number) for number in rounded_numbers]

    @staticmethod
    def strategy_permutations(strategies: Sequence[tuple[str, ...]]) -> list[tuple[str, ...]]:
        """Generate unique permutations of given strategies.

        Args:
            strategies (list): A list of compound strategies, where each strategy
                is a tuple of compounds.

        Returns:
            list: A sorted list of unique permutations of the input strategies.

        """
        unique_permutations = set()

        for strategy in strategies:
            for permutation in permutations(strategy):
                unique_permutations.add(permutation)

        return sorted(unique_permutations, key=len)

    @staticmethod
    def scale_values(numbers: list[int | float], target_number: int) -> list[float]:
        """Scales a list of numbers to a target sum while maintaining their relative proportions.

        This method performs a form of normalization or scaling. It takes a list
        of numbers and scales them so that their sum equals a specified target number, while 
        preserving the relative proportions between the original numbers.

        This is mainly a helper for smart_round(), but can be used in any context where you need 
        to adjust a set of values to sum to a specific total.

        Args:
            numbers (list): The input list of numbers to be scaled.
            target_number (int): The desired sum of the scaled values.

        Returns:
            list: A list of scaled values that sum to the target number, maintaining
                         the original proportions of the input numbers.

        Example:
            scale_values_by_contribution([10, 20, 30], 120)
            [24.0, 48.0, 72.0]
        """

        total_sum = sum(numbers)
        # Calculate how much I need to scale the values

        scaling_factor = target_number / total_sum
        # Apply the scale factor to each number
        distribution = [number * scaling_factor for number in numbers]

        return distribution

    def _transform_optimal_dist_to_expressions(self, distribution_row: list):
        """Transform the optimal distribution row into a list of compound-specific expressions.

        This method unpacks the compounds and laps from the optimal distribution,
        retrieves the corresponding tyre model for each compound, converts SymPy
        expressions to NumPy-friendly functions, and rounds the lap numbers.

        Args:
            distribution_row (list): A sequence of tuples, each containing 
                a compound (str) and its corresponding lap count (float).

        Returns:
            list: A list of tuples, each containing 
                (compound, lap_time_function, rounded_laps).
        """
        compounds, laps = zip(*distribution_row)
        compound_models = []

        for compound in compounds:
            # Retrieve the tyre model for the current compound
            expression = self._tyre_models[compound]['model']
            # Convert the SymPy expression to a NumPy-friendly function
            lap_time_function = sp.lambdify(self.lap_symbol, expression, "numpy")
            compound_models.append(lap_time_function)

        # Round the lap numbers to ensure they sum to the total race laps
        rounded_laps = self.smart_round(laps, self.num_laps)


        # Combine compounds, models, and rounded lap numbers
        transformed_distribution = list(zip(compounds, compound_models, rounded_laps))

        return transformed_distribution

    def _eval_expressions(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates expressions to create cumulative laptimes for each strategy.

        This method processes a DataFrame of expressions, each representing a race strategy.
        It calculates lap times for each stint in the strategy, applies pit losses between
        stints, and computes cumulative lap times for the entire race.

        Args:
            expression_df (pd.DataFrame): DataFrame containing expressions to evaluate.
                Each row represents a strategy with compound names, lap time functions,
                and number of laps for each stint.

        Returns:
            pd.DataFrame: A DataFrame where each column represents a strategy.
                Column names are compound sequences, and values are cumulative lap times.

        """
        data = {}
        expr_dict = expression_df.to_dict()
        for _, expr in expr_dict.items():
            laptimes = []
            compounds, functions, laps = zip(*expr)
            models = list(zip(functions, laps))

            for index, model in enumerate(models):
                func, lap_num = model
                # Generate an array of lap numbers for the current stint
                laps_run_on_compound = np.arange(1, lap_num + 1)
                # Calculate lap times for the current stint
                laptimes_on_compound = func(laps_run_on_compound)

                if index != len(models) - 1:
                    # Add pit loss to the last lap of the current stint
                    # This is not done for the last stint of the strategy
                    laptimes_on_compound[-1] += self.pit_loss

                laptimes.extend(laptimes_on_compound)

            # Create a strategy name by joining compound names
            strategy = ",".join(compounds)
            data[strategy] = np.cumsum(laptimes)

        # Convert the dictionary of cumulative lap times into a DataFrame
        return pd.DataFrame(data)

    def _create_strategy_performance_df(self, number_of_strategies: int) -> pd.DataFrame:
        """Create a DataFrame containing performance data for the best strategies.

        This method finds the N lowest (fastest) strategies, transforms them into
        expressions, evaluates these expressions to get lap times, and includes
        the fastest strategy's cumulative lap times for comparison.

        Args:
            number_of_strategies (int): The number of top strategies to include.

        Returns:
            pd.DataFrame: A DataFrame where each column represents a strategy's
                cumulative lap times, including the fastest strategy.

        Notes:
            The 'WinnerCumulativeTime' column represents the fastest strategy's
            cumulative lap times based on the theoretical optimal distribution,
            which may include fractional laps. This baseline is slightly faster
            than the same strategy with rounded lap numbers due to the optimization
            allowing for decimal lap counts (e.g., 22.3 laps on soft, 32.7 on medium).
            The plotted strategies use rounded lap numbers, which may result in
            slightly slower times even for the "fastest" strategy when compared
            to its theoretical optimal time.
        """
        # Find the N fastest strategies
        strat_df = self.find_n_lowest_strategies(number_of_strategies)

        # Transform each strategy's optimal distribution into expressions
        # This converts compound names and lap counts into lap time functions and rounded lap numbers
        expr_df = strat_df['optimal_dist'].apply(self._transform_optimal_dist_to_expressions)

        # Evaluate the expressions to get cumulative lap times for each strategy
        # This calculates lap times for each stint, applies pit losses, and computes cumulative times
        laptimes_df = self._eval_expressions(expr_df)

        # Calculate the cumulative lap times for the theoretical fastest strategy
        # This uses the optimal (potentially fractional) lap distribution
        fastest_strat = self._cumsum_of_avg_lap_time_of_fastest_strategy()

        # Add the fastest strategy's cumulative times to the DataFrame
        # This serves as a baseline for comparison with other strategies
        laptimes_df['WinnerCumulativeTime'] = fastest_strat

        return laptimes_df

    def get_lambda_tyre_models_expr(self) -> dict[str, Callable]:
        """Get lambdified expressions of tyre models for fast computation.

        This method converts the symbolic tyre model expressions into lambda
        functions that can be quickly evaluated. This is particularly useful
        for performance optimization when many evaluations of the tyre models
        are needed.

        Returns:
            dict: A dictionary mapping tyre compounds to their lambdified model
                expressions. These expressions can be called with lap numbers
                to quickly compute lap times.
        """
        models = {}
        for key, value in self._tyre_models.items():
            models[key] = sp.lambdify(self.lap_symbol, value['model'], 'numpy')
        return models

    def filter_by_stop(self, amount_of_stops: int, optional_df: pd.DataFrame = None) -> pd.DataFrame:
        """Filter strategies by the number of stops.

        This method creates a new DataFrame containing only the strategies with
        the specified number of pit stops. It can operate on either the main
        strategy DataFrame or an optionally provided DataFrame.

        Args:
            amount_of_stops (int): The number of stops to filter by.
            optional_df (pd.DataFrame, optional): DataFrame to filter. If None,
                                                uses self._strategy_df. 
                                                Defaults to None.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only strategies with the
                        specified number of stops.

        Raises:
            ValueError: If amount_of_stops is greater than the maximum number of
                        stops considered during the object's initialization.
        """

        if amount_of_stops > self._number_of_stops:
            raise ValueError(f'Index out of range. You are attempting to find {amount_of_stops} stop strategy but '
                             f'during initialization you entered a max of {self._number_of_stops} stops to consider ')

        df_to_use = optional_df if optional_df is not None else self._strategy_df

        # Creating a filter to show True for strategies that are the correct amount of stops
        n_stop_filter = df_to_use['stops'] == amount_of_stops

        return df_to_use[n_stop_filter]

    @staticmethod
    def get_racetime_and_optimal_dist(row: pd.Series):
        """Extract race time and optimal lap distribution from a DataFrame row.

        This helper method takes a single row from the strategy DataFrame and
        extracts the race time and optimal lap distribution. It's typically used
        to format the output of optimization methods.

        Args:
            row (pd.Series): A row from the strategy DataFrame, must contain
                            'racetime' and 'optimal_dist' fields.

        Returns:
            tuple: A tuple containing two elements:
                - float: The total race time.
                - tuple: The optimal lap distribution, where each element is a
                            tuple of (compound, number of laps).
        """

        # Extracting 'racetime' and 'optimal_dist' into a tuple
        race_time = row['racetime']
        optimal_dist = row['optimal_dist']

        # Creating the final tuple with the race time first, followed by the tyre distribution
        result_tuple = (race_time,) + tuple(optimal_dist)

        return result_tuple

    def _check_valid_starting_compound(self, compound: Optional[str] = None):
        """Validate the starting tyre compound.

        This method checks if the provided starting compound is valid for the current
        race strategy. It serves as a wrapper around the check_compound_validity method,
        with additional handling for None input.

        Args:
            compound (str or None): The tyre compound to validate. Can be None.

        Returns:
            str or None: The validated compound if it's valid and not None, 
                        or None if the input is None.

        Raises:
            ValueError: If the compound is not None but is not a valid tyre compound
                        according to the current tyre models.

        Note:
            This method is typically used during the initialization of the StrategyCombinatorics
            object to ensure that the starting compound, if specified, is valid.
        """

        if compound is not None:
            return self.check_compound_validity(compound)

        return None

    def _validate_available_tyres(self, available_tyres: list[tuple[str, int]] | None) \
            -> list[tuple[str, int]] | None:
        """Validate the list of available tyres for race strategy.

        This method performs several checks on the provided list of available tyres:
        1. Validates the basic structure and content of the tyre data.
        2. Ensures the starting compound (if specified) is present in the available tyres.
        3. Verifies that the number of planned stops is feasible given the available tyres.
        4. Checks compliance with regulations requiring at least two different compounds.

        Args:
            available_tyres (Optional[list of tuple]): A list where each tuple contains two elements:
                - compound (str): The tyre compound name.
                - laps (int): The number of laps already run on that compound (can be 0 - meaning new).
                Can be None if no tyres are specified.

        Returns:
            Optional[list of tuple]: The validated list of available tyres, or None if input was None.

        Raises:
            ValueError: If any of the following conditions are not met:
                - The starting compound is not in the available tyres list.
                - The number of planned stops exceeds the maximum possible stops.
                - Only one compound type is available (violating regulations).
            Any exceptions raised by the validate_tyre_data method.
        """
        if not available_tyres:
            return None

        if len(set(compound for compound, _ in available_tyres)) == 1:
            raise ValueError(f"Regulations require the use of at least two different compounds during the race. "
                             f"You have only entered one for the available tyres")

        self._validate_tyre_data(available_tyres, allow_zero_laps=True, context='available tyres')

        if self._start_compound:
            if all(self._start_compound != compound for compound, _ in available_tyres):
                raise ValueError("Data Mismatch: The starting compound is not present in the available tyres list.")

        max_number_stops_capable = len(available_tyres) - 1

        if self._number_of_stops > max_number_stops_capable:
            raise ValueError(f"The number of stops ({self._number_of_stops}) exceeds the maximum possible "
                             f"stops ({max_number_stops_capable}) based on available tyres.")

        return available_tyres

    def plot_strategy_performance(self, number_of_strategies: int = 3):
        """Plot the performance of the top strategies compared to the fastest strategy.

        This method creates a plot showing the time difference between each of the
        top N strategies and the fastest strategy over the course of the race.

        Args:
            number_of_strategies (int, optional): The number of top strategies to plot.
                Defaults to 3.

        Returns:
            None

        Displays:
            A Plotly interactive plot and a static matplotlib plot showing strategy
            performance over time.

        Notes:
            The plot shows the time difference on the y-axis (inverted) against
            the lap number on the x-axis. The fastest strategy serves as the
            baseline (0 seconds difference).

            The baseline (fastest strategy) is based on the theoretical optimal
            distribution, which may include fractional laps. This makes it slightly
            faster than the same strategy with rounded lap numbers. For example,
            a theoretical optimal strategy might use 22.3 laps on soft and 32.7 on
            medium, while the plotted version would round to 22 and 33 laps
            respectively. This rounding can result in a slightly slower time, even
            for the "fastest" strategy, when compared to its theoretical optimal time.
            This discrepancy is due to the optimization process allowing for decimal
            lap counts, which isn't possible in actual race conditions.
        """
        strat_performance_df = self._create_strategy_performance_df(number_of_strategies)

        # Create a figure and axis for the plot
        fig = go.Figure()

        laps = np.arange(1, self.num_laps + 1)

        # Plot each driver's time difference with the winner's cumulative time (ie the fastest)
        for column in strat_performance_df.columns:
            if column != 'WinnerCumulativeTime':  # Avoid plotting the winner against itself
                # Calculate the difference between the driver's times and the winner's times
                time_difference = strat_performance_df[column] - strat_performance_df['WinnerCumulativeTime']
                fig.add_trace(go.Scatter(x=laps,
                                         y=time_difference,
                                         name=column,
                                         mode='lines+markers'
                                         ))

        # Edit the layout
        fig.update_layout(title='Strategy Comparison (Clean Air Optimisation)',
                          xaxis_title='Lap number',

                          yaxis_title='Time Difference (s)',
                          legend_title='Strategy (compounds used)',
                          legend=dict(font=dict(size=22))


                          )

        fig.update_layout(yaxis_autorange="reversed")

        print("""
        Note: Baseline uses optimal fractional (float) laps. Plotted strategies use rounded (int/discrete) laps,
        always appearing slower. The strategy closest to the baseline is equivalent to the baseline it's plotted 
        against, considering this discrepancy, it is the fastest theoretical strategy in clean-air conditions.
            """)

        fig.show()

    def plot_tyre_performance(self):
        """
        Plot that shows the distinct tyre performance of the models over a race distance
        """
        # Create traces
        fig = go.Figure()

        laps = np.arange(1, self.num_laps + 1)
        tyre_models = self.get_lambda_tyre_models_expr()

        colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'white', }

        # not used in modern f1 just thought id put them for fun
        old_colors = {'hyper_soft': 'pink',
                      'ultra_soft': 'purple',
                      'super_soft': 'red',
                      'soft': 'yellow',
                      'medium': 'white',
                      'hard': 'blue',
                      'super_hard': 'orange',
                      }

        if any(model not in colors for model in tyre_models):
            colors = old_colors

        for model_name, expression in tyre_models.items():
            laptimes = expression(laps)

            # Check if the model_name has a specified color, otherwise let Matplotlib choose the color
            if model_name in colors:
                fig.add_trace(go.Scatter(x=laps, y=laptimes, mode='lines+markers', name=model_name,
                                         line=dict(color=colors[model_name], )))
            else:
                fig.add_trace(go.Scatter(x=laps, y=laptimes, mode='lines+markers', name=model_name,))
        # Edit the layout
        fig.update_layout(title='Tyre Performance Comparison',
                          xaxis_title='Lap number',
                          yaxis_title='lap time (s)',
                          legend_title='Tyres',
                          legend=dict(font=dict(size=22))
                          )
        fig.show()
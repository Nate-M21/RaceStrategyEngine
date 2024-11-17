"""This module implements the MonteCarloRaceSimulations class, which extends the RaceStrategyEngine class
to perform Monte Carlo simulations for exploring various race strategies and outcomes in Formula 1.

The MonteCarloRaceSimulations class provides functionality for:
- Running multiple race simulations with varying strategies and pit stop timings
- Analyzing probabilistic race results based on different strategic variations
- Visualizing simulation results through various plots and charts
This class is particularly useful for assessing the effectiveness of different race strategies
under various conditions and for understanding the range of possible outcomes in a race."""

from itertools import zip_longest
import copy
import random
from functools import cache
from typing import Literal
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType, Row
from tqdm import tqdm

from .driver import Driver
from .monte_carlo_analysis import plot_drivers_finishing_position_frequency, plot_traffic_heatmap, \
    plot_traffic_boxplot, plot_race_strategy_position_distribution, plot_drivers_position_dominance, \
    plot_race_strategy_ranking
from .race_configuration import RaceConfiguration
from .combinatorics import StrategyCombinatorics
from .race_simulation import RaceSimulation
from .utility import safe_get_from_list, time_simulation, count_driver_simulations



class MonteCarloRaceSimulations(RaceSimulation):
    """
    Extends RaceStrategyEngine to implement Monte Carlo methods for exploring various race strategies
    and outcomes. This class facilitates the running of multiple simulations to analyze probabilistic
    race results based on different strategic variations and pit stop timings.
    """

    def __init__(self, drivers: dict[str, Driver], starting_race_grid: dict, race_configuration: RaceConfiguration,
                 lap_variation_range: tuple[int, int] = (-10, 10)):
        super().__init__(drivers, starting_race_grid, race_configuration)

        if not (isinstance(lap_variation_range, tuple) and len(lap_variation_range) == 2 and
                all(isinstance(x, int) for x in lap_variation_range)):
            raise ValueError("lap_variation_range must be a tuple of two integers")

        self._lap_variation_range = lap_variation_range
        # Alternatives strategy choices are pre-computed within the Driver object with the aid of the
        # Combinatorics class to save CPU from calculating this every time.
        self.drivers_alternate_strategies = self._get_strategy_alternatives()
        self.drivers_max_stops_alternate_strategies = self._get_drivers_max_stops_in_alternative_strategies()
        self._max_stops_among_all_the_drivers = self._get_max_stop_among_all_drivers()

        # Precompute strategies by stops for each driver
        self.drivers_strategies_by_stops = {
            driver: self.drivers[driver].get_alternate_strategies(dict_mode=True)
            for driver in self.drivers
        }
        # For schema purposes, how the dataframe's columns will be ordered
        self._schema_order = [
            ('driver', StringType()),
            ('compounds_used', ArrayType(StringType())),
            ('pit_stop_laps', IntegerType()),
            ('points', IntegerType()),
            ('position', IntegerType()),
            ('start_position', IntegerType()),
            ('stops', IntegerType()),
            ('tyre_usage', ArrayType(ArrayType(StringType()))),
            ('race_time', FloatType()),
            ('laps_behind_traffic', IntegerType()),

        ]
        # The single highest number of pit stops found in their alternate strategies
        # Setting the default to 1
        self._driver_max_stop = 1

        # Find the index where pit stop fields start and end, this for future proofing if I decide to change the
        # order of fields so the rest of my code doesn't break
        self._pit_stops_start_idx = self._schema_order.index(('pit_stop_laps', IntegerType()))
        self._pit_stop_end_idx = self._pit_stops_start_idx + self._max_stops_among_all_the_drivers

        self._drivers_pit_stop_indices = self._get_drivers_pit_stop_index()

        # For the explicit schema, where the exact number of columns, their order, and their names will be stored
        _, self._schema_fields = self._structure_generator(self._max_stops_among_all_the_drivers)

    @property
    def schema(self):
        """Get the Spark SQL schema for the simulation results.

        This property returns the Spark SQL schema that defines the structure of the simulation results.
        The schema is generated based on the maximum number of stops among all drivers 

        Returns:
            StructType: The Spark SQL schema for the simulation results.

        """

        return self._generate_schema()

    def _get_drivers_max_stops_in_alternative_strategies(self) -> dict:
        """
        Get the longest strategy each driver has in their alternative strategies list.

        Returns:
            (dict): A dictionary mapping driver names to their maximum number of stops in alternative strategies.
        """

        return {name: driver_object.highest_amount_of_stops_in_alternate_strategies for name, driver_object in
                self.drivers.items()}

    def _get_max_stop_among_all_drivers(self) -> int:
        """ Get the maximum stop amongst all drivers

        This method is a helper in defining the schema, it finds the max amount of stops that will
        be performed, and uses it to define the schema pit fields

        Returns:
            (int): An int representing the most stops that can occur in the simulation


        """
        most_stops = max(self.drivers_max_stops_alternate_strategies.values())
        return most_stops

    def _update_driver_strategy_dependent_attributes(self):
        """Recalculate all strategy-dependent attributes for all drivers.

        This method updates the following attributes:
        - driver_strategies: The starting strategies for each driver.
        - lambdified_drivers_tyre_models: Lambdified versions of tyre models for quick evaluation.
        - drivers_tyre_models: Current tyre models for each driver.
        - drivers_alternate_strategies: Alternative strategies for each driver.
        - drivers_max_stops_alternate_strategies: Maximum number of stops in alternative strategies.

        This method should be called after any changes that affect driver strategies or tyre models.
        It extends the parent class method to include Monte Carlo specific attributes.
        """
        super()._update_driver_strategy_dependent_attributes()

        self.drivers_alternate_strategies = self._get_strategy_alternatives()
        self.drivers_max_stops_alternate_strategies = self._get_drivers_max_stops_in_alternative_strategies()
        self._max_stops_among_all_the_drivers = self._get_max_stop_among_all_drivers()

        # Heavy
        self.drivers_strategies_by_stops = {
            driver: self.drivers[driver].get_alternate_strategies(dict_mode=True)
            for driver in self.drivers
        }

    def _get_strategy_alternatives(self) -> dict:
        """
        Retrieve alternate strategy choices for all drivers and organize them into a dictionary.

        Returns:
            (dict): A dictionary mapping driver names to their list of alternative strategies.
        """

        return {name: driver_object.get_alternate_strategies() for name, driver_object in self.drivers.items()}

    def _select_random_strategy(self, driver):
        """Randomly select and modify a strategy for a driver.

        This method selects an alternate strategy for the given driver, shuffles the order
        of the compounds, and modifies the number of laps for each stint within a predefined
        range. The resulting modified strategy is then assigned to the driver.

        Args:
            driver (str): The name of the driver for whom to select a random strategy.
        """

        alternate_strategy = self._select_alternate_strategy(driver)

        shuffled_strategy = self._shuffle_strategy(alternate_strategy)

        modified_strategy = self._modify_strategy(driver=driver, strategy=shuffled_strategy,
                                                  lap_variation_range=self._lap_variation_range)

        self.driver_strategies[driver] = modified_strategy

    def _select_alternate_strategy(self, driver):
        """Select an alternate strategy for a driver.

        This method randomly selects an alternate strategy for the given driver from their
        precomputed strategies. It first chooses a random number of stops and then selects
        a strategy from the available options for that stop count.

        Args:
            driver (str): The name of the driver for whom to select an alternate strategy.

        Returns:
            tuple: The selected alternate strategy as a tuple of (race_time, compound1, laps1, compound2, laps2, ...).
        """

        strategies = self.drivers_strategies_by_stops[driver]

        # Select a stop count with equal probability
        stop_count = random.choice(list(strategies.keys()))

        # Select a strategy within the chosen stop count
        selected_strategy = random.choice(strategies[stop_count])

        return selected_strategy

    def _shuffle_strategy(self, strategy):
        """Shuffle the order of the compounds in a strategy.

        This method takes a strategy tuple and shuffles the order of the compounds while
        keeping the number of laps for each stint unchanged. It returns a new strategy tuple
        with the shuffled compound order.

        Args:
            strategy (tuple): The strategy to shuffle, in the format (race_time, compound1, laps1, compound2, laps2,...).

        Returns:
            tuple: A new strategy tuple with the compounds shuffled.
        """

        total_time, *compound_strategy = strategy
        # Shuffle the order in which the tyres will be used
        random.shuffle(compound_strategy)

        shuffled_strategy = tuple((total_time, *compound_strategy))

        return shuffled_strategy

    def _modify_strategy(self, driver, strategy, lap_variation_range):
        """Modify the number of laps for each stint in a strategy.

        This method takes a strategy tuple and modifies the number of laps for each stint
        by a random amount within the specified lap_variation_range. It returns a new strategy
        tuple with the modified lap counts.

        Args:
            driver (str): The name of driver which strategy is being modified
            strategy (tuple): The strategy to modify, in the format (race_time, compound1, laps1, compound2, laps2, ...).
            lap_variation_range (tuple): A tuple specifying the range of lap variation (min, max).

        Returns:
            tuple: A new strategy tuple with the modified lap counts.
        """

        total_race_time, *compound_strategy = strategy
        compounds, laps_on_compounds = zip(*compound_strategy)

        # Randomly adjust the number of laps for each compound
        modified_laps = self._adjust_laps_randomly(list(laps_on_compounds), lap_variation_range)

        # Construct the new modified strategy
        modified_lap_dist = tuple(zip(compounds, modified_laps))
        driver_object = self.drivers[driver]
        new_strategy = driver_object.strategy_creator(new_lap_distributions=modified_lap_dist)

        return new_strategy

    def _adjust_laps_randomly(self, laps_on_compounds, lap_variation_range):
        """Adjust the number of laps for each stint in a strategy.

        This method takes a list of lap counts for each stint and adjusts them by a random
        amount within the specified lap_variation_range. It ensures that the total number of
        laps across all stints matches the race distance and that each stint has at least one lap.

        Args:
            laps_on_compounds (list): A list of lap counts for each stint.
            lap_variation_range (tuple): A tuple specifying the range of lap variation (min, max).

        Returns:
            list: A new list of adjusted lap counts for each stint.
        """
        modified_laps = [max(1, laps + random.randint(*lap_variation_range)) for laps in laps_on_compounds]

        # Check if any compound is assigned less than 1 lap or total laps doesn't equal number of race laps
        if any(lap < 1 for lap in modified_laps) or sum(modified_laps) != self.race_config.num_laps:
            # applying smart round to fix the error
            modified_laps = StrategyCombinatorics.smart_round(modified_laps,
                                                              self.race_config.num_laps,
                                                              scale_by_contribution=True)

        return modified_laps  # Return the modified laps when valid

    def run_adaptive_monte_carlo_simulation(self):
        """Run an adaptive Monte Carlo simulation.

        This method runs a Monte Carlo simulation with adaptive constraints, considering the
        current race state. It performs the simulation for all drivers
        and returns the results as a list of Spark Row objects.

        Returns:
            List[spark_row]: A list of Spark Row objects containing the simulation results.
        """
        all_drivers = set(list(self._race_state.keys()))
        return self._mc_simulation(selected_drivers_to_analyse=all_drivers, adaptive_constraints=True)

    def _mc_simulation(self, selected_drivers_to_analyse: set, adaptive_constraints=False):
        """Run a Monte Carlo simulation for the selected drivers.

        This method runs a single iteration of the Monte Carlo simulation for the specified
        drivers. It applies the selected strategies, runs the simulation, and collects the
        results in a structured format compatible with Spark DataFrames.

        Args:
            selected_drivers_to_analyse (set): A set of driver names to include in the simulation.
            adaptive_constraints (bool, optional): If True, applies adaptive constraints based on the
                current race state when selecting strategies. Defaults to False.

        Returns:
            list: A list of Spark Row objects containing the simulation results for each driver.
        """

        # Run the monte carlo simulation initially
        self.run_monte_carlo_simulation(adaptive=adaptive_constraints)

        fields = self._generate_field_names()
        simulation_results = []

        race_results: dict = self.get_result_from_simulation_run(format_json=False)

        # Process only drivers who didn't retire and are selected for analysis
        for selected_driver in set(self._sim_race_state.keys()) & selected_drivers_to_analyse:
            # TODO fix this mess i made, there is cleaner way
            driver_results: dict = race_results[selected_driver]
            pit_laps = driver_results['pit_laps']
            compounds_used = driver_results['strategy']
            laps_behind_traffic = driver_results['laps_behind_traffic']
            result = driver_results['end_position']
            starting_position = driver_results['start_position']
            points = driver_results['points']

            amount_of_stops = driver_results['stops']
            tyre_usage = driver_results['tyre_usage']
            race_time = driver_results['race_time']

            row_data = {
                'driver': selected_driver,
                'compounds_used': compounds_used,
            }
            for i, field in enumerate(fields[self._pit_stops_start_idx:self._pit_stop_end_idx]):
                row_data[field] = int(safe_get_from_list(pit_laps, i, default=0))
            row_data['points'] = points
            row_data['position'] = result

            row_data['start_position'] = starting_position
            row_data['stops'] = amount_of_stops
            row_data['tyre_usage'] = tyre_usage
            row_data['race_time'] = race_time
            row_data['laps_behind_traffic'] = laps_behind_traffic

            if self._valid_simulation_result_structure(tuple(self._schema_fields), tuple(row_data)):
                # Create a Row object for each simulation result to structure the data appropriately for Spark DataFrame
                # operations. This allows for efficient manipulation and analysis of the simulation data across the
                # distributed system. if you want to add extra information to Row object you need to first add into the
                # self._schema_order, so it is added to the schema
                simulation_results.append(Row(**row_data))

        return simulation_results

    @cache
    def _valid_simulation_result_structure(self, schema: tuple, simulation_result_keys: tuple) -> bool:
        """Validates that the structure of the simulation result (named tuple) matches the expected schema.

        This method ensures that all fields defined in the schema are present in the simulation result,
        and that there are no extra fields in the simulation result that are not defined in the schema.

        Args:
            schema (tuple): A tuple of (field_name, field_type) pairs representing the expected
                        structure of the simulation result. Typically created from a list
                        (e.g., tuple(self._schema_fields)).
            simulation_result_keys (tuple): A tuple of the keys from the simulation result dictionary.
                                        Created directly from the dictionary (e.g., tuple(result_dict)).

        Raises:
            ValueError: If there's a mismatch between the schema and the simulation result.
                        The error message provides detailed information about missing or
                        extra fields, as well as the order and types of fields in both
                        the schema and the simulation result.

        Returns:
            bool: True if the simulation result structure matches the schema, False otherwise.
                Note that in the current implementation, False is never explicitly returned
                as a ValueError is raised in case of a mismatch.

        Notes:
            - The result is cached so that repeated calls in the Monte Carlo simulations don't have
            to recalculate the same thing with identical inputs, improving performance.
            - For the 'compounds_used' key in the simulation namedtuple, which is typically a tuple or list,
            the corresponding type ArrayType(StringType()) in the schema is a sufficient type.
            - The 'schema' parameter is expected to be a tuple, typically created from a list (e.g., 
            tuple(self._schema_fields)). This allows for caching while maintaining mutability of 
            the original schema list in the class. The cache key uses the immutable tuple, enabling 
            efficient lookups for repeated schema structures.
        """

        field_names, _ = zip(*schema)

        set_fields = set(field_names)
        set_simulation_row = set(simulation_result_keys)
        if set_fields != set_simulation_row:

            missing_in_fields = set_simulation_row - set_fields
            missing_in_simulation = set_fields - set_simulation_row

            raise ValueError("Data mismatch between the schema and the simulation result\n"
                             f"Fields missing from schema: {missing_in_fields or None}\n"
                             f"Fields missing from the simulation: {missing_in_simulation or None}\n"
                             f"Extra Information - Ensure all fields are present and types match the schema. \n\n"
                             f"Fields in the Simulation:\n{'-' * 30}\n {simulation_result_keys}\n"
                             f"\n"
                             f"Fields in the Schema:\n{'-' * 30}\n {field_names}\n"
                             f"\n"
                             f"Schema field types (name, type):\n{'-' * 30}\n {schema}\n"
                             f"\nNote: For the 'compounds_used' field, which is typically a tuple or list, "
                             f"the type ArrayType(StringType()) in the schema is sufficient."
                             )
        else:  # Just being explicit with the else
            return True

    @cache
    def _generate_schema(self):
        """
        Generates a Spark SQL schema for the simulation results.

        This method creates a StructType schema that matches the structure of the simulation results.
        It uses the schema fields prepared by _generate_field_names to ensure consistency.

        Returns:
            StructType: A Spark SQL schema definition for the simulation results.
        """

        _, schema_fields = self._structure_generator(self._max_stops_among_all_the_drivers)

        # noinspection PyTypeChecker
        return StructType([StructField(*schema_field, True) for schema_field in schema_fields])

    def _generate_field_names(self):
        """
        Generates a list of field names for the simulation results based on the driver's maximum number of pit stops.
        
        This method creates a consistent set of field names used both for creating the schema and
        structuring the simulation results. It includes 'compounds_used', a dynamic number of pit lap fields,
        and 'points'.

        Returns:
            list: A list of field names as strings, including 'compounds_used', pit lap fields, and 'points'.
        """
        max_amount_of_stops = self._max_stops_among_all_the_drivers

        field_names, _ = self._structure_generator(max_amount_of_stops)
        return field_names

    def _get_drivers_pit_stop_index(self):
        """Get the pit stop field indices for each driver.

        This method calculates the start and end indices of the pit stop fields in the
        simulation results schema for each driver based on their maximum number of stops.

        Returns:
            dict: A dictionary mapping driver names to tuples of (start_index, end_index) for
                their pit stop fields in the schema.
        """
        driver_pit_index = {}
        for driver in self.drivers:
            driver_max_amount_of_stops = self.drivers_max_stops_alternate_strategies[driver]

            driver_pit_stops_start_idx = self._schema_order.index(('pit_stop_laps', IntegerType()))
            driver_pit_stop_end_idx = driver_pit_stops_start_idx + driver_max_amount_of_stops

            driver_pit_index[driver] = (driver_pit_stops_start_idx, driver_pit_stop_end_idx)

        return driver_pit_index

    def _structure_generator(self, max_amount_of_stops: int):
        """Generate the field names and schema structure for the simulation results.

        This method creates the field names and schema structure for the simulation results
        based on the maximum number of stops. It includes fields for the driver name, compounds
        used, pit stop laps, points, position, and other relevant information.

        Args:
            max_amount_of_stops (int): The maximum number of stops to consider in the schema.

        Returns:
            tuple: A tuple containing two elements:
                - fields (list): A list of field names for the simulation results.
                - schema_fields (tuple): A tuple of (field_name, field_type) pairs defining the schema structure.
        """
        fields = []
        schema_fields = []
        for field, field_type in self._schema_order:
            if field == 'pit_stop_laps':
                for i in range(1, max_amount_of_stops + 1):
                    pit_lap_keys = f'pit_stop_{i}_lap'
                    fields.append(pit_lap_keys)
                    schema_fields.append((pit_lap_keys, field_type))
            else:
                fields.append(field)
                schema_fields.append((field, field_type))

        return fields, tuple(schema_fields)

    def run_monte_carlo_simulation(self, adaptive=False):
        """Execute a Monte Carlo simulation of a race.

        This method orchestrates a Monte Carlo simulation of a race by iteratively modifying
        each driver's strategy, shuffling it, and adjusting it based on a predefined range
        of lap variations. After all modifications, it triggers the actual race simulation.

        Args:
            adaptive (bool, optional): If True, the simulation uses adaptive constraints based
                on the current race state. Defaults to False.
        """

        for driver in self._race_state:
            if adaptive:
                new_strategy = self._generate_adaptive_strategy(driver)

                self.driver_strategies[driver] = new_strategy
            else:
                self._select_random_strategy(driver)

        self.run_simulation()

    @cache
    def _generate_combinations(self, driver: str, filter_compounds_by: tuple, minimum_stops: int = 1,
                               output: Literal['DICTIONARY', 'LIST'] = 'DICTIONARY'):
        """Generate valid strategy combinations for a driver.

        This method generates all valid strategy combinations for the given driver, filtered
        by the specified compounds and minimum number of stops. The combinations can be returned
        as a list or a dictionary grouped by the number of stops.

        Args:
            driver (str): The name of the driver for whom to generate combinations.
            filter_compounds_by (tuple): A tuple of compounds to include in the combinations.
            minimum_stops (int, optional): The minimum number of stops to consider. Defaults to 1.
            output (Literal['DICTIONARY', 'LIST'], optional): The format of the output. Defaults to 'DICTIONARY'.
                - 'DICTIONARY': Returns a dictionary mapping the number of stops to a list of combinations.
                - 'LIST': Returns a flat list of all combinations.

        Returns:
            dict or list: The generated strategy combinations in the specified format.
        """
        if output not in ['DICTIONARY', 'LIST']:
            raise ValueError('The output must be either LIST or DICTIONARY')

        driver_object = self.drivers[driver]

        combinations = driver_object.strategy.generate_combinations(filter_strategy_by=filter_compounds_by,
                                                                    minimum_stops=minimum_stops)

        if output == 'LIST':
            return combinations

        combination_by_stop = defaultdict(list)

        for combination in combinations:
            amount_of_stops = len(combination) - 1
            combination_by_stop[f"{amount_of_stops}_stop"].append(combination)

        return combination_by_stop

    def _fetch_random_combination(self, driver: str, required_compounds: list, minimum_stops: int = 1):
        """Fetch a random valid strategy combination for a driver.

        This method selects a random strategy combination for the given driver that includes
        the required compounds and has at least the specified minimum number of stops.

        Args:
            driver (str): The name of the driver for whom to fetch a combination.
            required_compounds (list): A list of compounds that must be included in the combination.
            minimum_stops (int, optional): The minimum number of stops the combination must have. Defaults to 1.

        Returns:
            tuple: A randomly selected valid strategy combination.
        """
        # converting to tuple so the method below can cache it if it the same
        required_compounds = tuple(required_compounds)
        combinations = self._generate_combinations(driver, required_compounds, minimum_stops=minimum_stops)

        type_of_stop = random.choice(list(combinations.keys()))

        new_combination = random.choice(combinations[type_of_stop])

        return new_combination

    def _fetch_adaptive_lap_distribution(self, driver):
        """Generate an adaptive lap distribution for a driver based on their current race state.

        This method creates a context-aware lap distribution strategy for the given driver. It 
        considers the driver's used tyres, current tyre, and the remaining race distance to generate 
        a strategy that completes the race. The method adapts to different scenarios:
        1. If the current strategy will complete the race, it returns the current distribution.
        2. If the driver has already pitted, it adjusts the current stint to complete the race.
        3. If the driver hasn't pitted, it generates a valid one-stop strategy.

        Args:
            driver (str): The name of the driver for whom to generate the adaptive strategy.

        Returns:
            list: An adaptive lap distribution strategy as a list of (compound, laps) tuples.

        Note:
            This method is useful when the standard random strategy generation methods 
            cannot find a valid combination, ensuring a reasonable strategy is always available.
        """
        current_tyre_stint, used_tyres_stint, required_compounds, _ = self._get_current_driver_stint_info(driver)

        current_tyre, current_tyre_laps_age = current_tyre_stint
        used_tyres = used_tyres_stint
        total_used_tyre_laps = sum([laps for _, laps in used_tyres])

        strategy_will_finish_race = self.race_config.num_laps == (total_used_tyre_laps + current_tyre_laps_age)
        if strategy_will_finish_race:
            lap_distribution_to_end = []
            lap_distribution_to_end.extend(used_tyres)
            current_stint_to_the_end = (current_tyre, current_tyre_laps_age)
            lap_distribution_to_end.append(current_stint_to_the_end)

            return lap_distribution_to_end

        # They are going to end with their current strategy
        if used_tyres:  # if they have pitted
            lap_distribution = []

            current_stint_laps = self.race_config.num_laps - total_used_tyre_laps

            lap_distribution.extend(used_tyres)
            current_stint = (current_tyre, current_stint_laps)
            lap_distribution.append(current_stint)

            return lap_distribution

        else:  # if they have not pitted. i should use this when the combination being proposed is one stop
            required_compounds = tuple(required_compounds)

            combinations = self._generate_combinations(driver=driver, filter_compounds_by=required_compounds,
                                                       output='LIST')
            one_stop_combinations = [combination for combination in combinations if len(combination) == 2]
            one_stop_combination = random.choice(one_stop_combinations)
            _, last_compound = one_stop_combination

            current_stint_laps = max(1, self._race_state[driver]['current_tyre_laps_age'])
            current_stint = (current_tyre, current_stint_laps)

            last_stint_laps = self.race_config.num_laps - current_stint_laps
            last_stint = (last_compound, last_stint_laps)

            lap_distribution = [current_stint, last_stint]

            return lap_distribution

    def _get_current_driver_stint_info(self, driver):
        """Get the stint information for a driver based on current race state.

        This method retrieves the stint information for the given driver, including their
        used tyres, current tyre, and the required compounds and laps based on the race state.

        Args:
            driver (str): The name of the driver for whom to get the stint information.

        Returns:
            tuple: A tuple containing the following elements:
                - current_stint (list): The current stint as a list of [compound, laps].
                - used_stints (list): The used stints as a list of (compound, laps) tuples.
                - required_compounds (list): The compounds that are required to be used.
                - required_laps (list): The laps corresponding to each required compound.
        """
        used_stints: list = self._race_state[driver].get('used_tyres', [])

        required_compounds = []
        used_tyres_compounds = [compound for compound, _ in used_stints]
        required_compounds.extend(used_tyres_compounds)

        required_laps = []
        used_tyres_laps = [laps for _, laps in used_stints]
        required_laps.extend(used_tyres_laps)

        current_tyre_compound = self._race_state[driver]['current_tyre']
        current_tyre_laps_age = self._race_state[driver]['current_tyre_laps_age']
        if current_tyre_compound:  # If it not None, it shouldn't be but if it
            required_compounds.append(current_tyre_compound)
            required_laps.append(current_tyre_laps_age)

        current_stint = list((current_tyre_compound, current_tyre_laps_age))
        return current_stint, used_stints, required_compounds, required_laps

    def _generate_adaptive_strategy(self, driver: str):
        """Generate an adaptive strategy for a driver based on their current race state.

        This method creates an adaptive strategy for the given driver considering their used
        tyres, current tyre, and required compounds. It fetches a random valid combination and
        generates a lap distribution based on the constraints.

        Args:
            driver (str): The name of the driver for whom to generate an adaptive strategy.

        Returns:
            tuple: The generated adaptive strategy as a tuple.
        """

        _, used_stints, required_compounds, required_laps = self._get_current_driver_stint_info(driver)

        compounds = self._fetch_random_combination(driver=driver, required_compounds=required_compounds)

        if required_compounds:
            lap_distribution = self._generate_lap_distribution(driver=driver, strategy_compounds=compounds,
                                                               required_compounds=required_compounds,
                                                               required_laps=required_laps,
                                                               used_tyres=used_stints)

            new_strategy = self.drivers[driver].strategy_creator(new_lap_distributions=tuple(lap_distribution))

            return new_strategy

        else:

            return self._select_alternate_strategy(driver)

    def _generate_lap_distribution(self, driver, strategy_compounds, required_compounds, required_laps,
                                   used_tyres: list[tuple[str, int]]):
        """Generate the lap distribution for a driver based on a strategy and constraints.

        This method optimizes the lap distribution for the given driver based on the strategy
        compounds, required compounds, required laps, and used tyres. It separates the strategy
        into past/current stints and future stints, and adjusts the future stints based on the
        remaining laps and constraints.

        Args:
            driver (str): The name of the driver for whom to generate the lap distribution.
            strategy_compounds (list or tuple): The compounds of the chosen strategy.
            required_compounds (list or tuple): The compounds that are required to be used.
            required_laps (list): The laps corresponding to each required compound.
            used_tyres (list): The used tyres and their corresponding laps as a list of (compound, laps) tuples.

        Returns:
            list: The generated lap distribution as a list of (compound, laps) tuples.
        """

        preliminary_lap_distribution = list(zip_longest(strategy_compounds, required_laps, fillvalue=1))

        past_and_current_stints_amount = len(required_compounds)
        past_and_current_stints = preliminary_lap_distribution[:past_and_current_stints_amount]
        future_stints = preliminary_lap_distribution[past_and_current_stints_amount:]

        if future_stints:

            return self._adjust_current_and_future_stint_lengths(driver=driver,
                                                                 past_and_current_stints=past_and_current_stints,
                                                                 future_stints=future_stints,
                                                                 used_tyres=used_tyres)

        else:  # if there are no future stints there nothing left to, this tyre is going to the end test it
            return self._fetch_adaptive_lap_distribution(driver=driver)

    def _adjust_current_and_future_stint_lengths(self, driver, past_and_current_stints, future_stints, used_tyres):
        """Adjust the current and future stint lengths for a driver based on constraints and remaining laps.

        This method adjusts the stint lengths for the given driver, considering their past, current,
        and future stints, used tyres, and the remaining laps in the race. It ensures that
        the total number of laps matches the race distance and that each stint has a valid length.

        Args:
            driver (str): The name of the driver for whom to adjust the stint lengths.
            past_and_current_stints (list): The past and current stints as a list of (compound, laps) tuples.
            future_stints (list): The future stints as a list of (compound, laps) tuples.
            used_tyres (list): The used tyres and their corresponding laps as a list of (compound, laps) tuples.

        Returns:
            list: The adjusted lap distribution as a list of (compound, laps) tuples
        """

        total_used_tyre_laps = sum([laps for _, laps in used_tyres])
        current_tyre_laps_age = self._race_state[driver]['current_tyre_laps_age']
        race_laps_left = self.race_config.num_laps - (total_used_tyre_laps + current_tyre_laps_age)

        stints_to_randomize = []

        current_stint = past_and_current_stints[-1]  # the stint right now
        stints_to_randomize.append(current_stint)

        # shuffling to get different permutations of the future stints only
        random.shuffle(future_stints)
        stints_to_randomize.extend(future_stints)
        min_laps_required = len(stints_to_randomize)
        enough_laps_remain = race_laps_left >= min_laps_required

        if enough_laps_remain:
            new_lap_distribution = []

            # Just for easier readability 
            past_stints = used_tyres

            laps_to_use_for_randomizing = self.race_config.num_laps - total_used_tyre_laps
            compounds, stints_to_randomize_laps_to_run = self._extract_strategy_compounds_and_laps(
                driver, stints_to_randomize, included_race_time=False)

            new_laps = self._randomize_current_and_future_stint_lengths(stints_to_randomize_laps_to_run,
                                                                        current_tyre_laps_age,
                                                                        laps_to_use_for_randomizing,
                                                                        self._lap_variation_range)

            new_current_and_future_stints = list(zip(compounds, new_laps))
            new_lap_distribution.extend(past_stints)
            new_lap_distribution.extend(new_current_and_future_stints)

            return new_lap_distribution

        else:
            return self._fetch_adaptive_lap_distribution(driver=driver)

    @staticmethod
    def _randomize_current_and_future_stint_lengths(current_and_future_stints, current_tyre_laps, remaining_laps,
                                                    lap_variation_range):
        """Randomize the current and future stint lengths based on constraints and remaining laps.

        This method randomizes the stint lengths for the current and future stints, considering the current
        tyre age, remaining laps, and a lap variation range. It ensures that the total number
        of laps matches the remaining distance and that each stint has a valid length.

        Args:
            current_and_future_stints (list): The current and future stints as a list of laps.
            current_tyre_laps (int): The number of laps completed on the current tyre.
            remaining_laps (int): The remaining laps after accounting for used tyres.
            lap_variation_range (tuple): The range of lap variation (min, max) to apply.

        Returns:
            list: The randomized stint lengths as a list of laps.
        """

        new_laps = []
        amount_of_stints = len(current_and_future_stints)

        max_laps_to_run = remaining_laps - amount_of_stints + 1
        future_stints = current_and_future_stints[1:]

        # The + 1 accounts for the fact that driver is currently on a lap that hasn't been completed yet but will be
        # so effectively the min he can do on the current tyre is current_tyre_laps + 1 unless he retires
        min_lap_on_current_tyre = current_tyre_laps + 1
        current_stint_laps_final = max(1, random.randint(min_lap_on_current_tyre, max_laps_to_run))
        new_laps.append(current_stint_laps_final)

        laps_left_over_for_future_stints = remaining_laps - current_stint_laps_final
        random_future_stints = [max(1, laps + random.randint(*lap_variation_range)) for laps in future_stints]
        future_stints_laps_final = StrategyCombinatorics.smart_round(numbers=random_future_stints,
                                                                     target_sum=laps_left_over_for_future_stints,
                                                                     scale_by_contribution=True)
        new_laps.extend(future_stints_laps_final)

        return new_laps

    def _check_simulation_run(self, custom_message=None):
        message = ("Simulation state is not initialized. Please run the simulation first by utilizing the"
                   " 'run_monte_carlo_simulation()' method\n. If you would like to avoid the stochastic effects"
                   " of the monte carlo you may use 'run_simulation()' instead.")
        super()._check_simulation_run(custom_message=message)

    def run_and_analyze_simulations_from_monte_carlo_runs(self, drivers_to_analyse: str | list[str] | Literal['ALL'],
                                                          num_simulations: int,
                                                          method: Literal['single-core', 'multi-core',] = 'multi-core',
                                                          immediate_plot: bool = False
                                                          ):

        """Run Monte Carlo simulations and analyze results for specified driver(s).

        This method runs a specified number of Monte Carlo simulations for the selected driver(s)
        using either single-core or multi-core processing. It then analyzes the simulation results
        and optionally generates plots immediately after the simulations.

        Args:
            drivers_to_analyse (str | List[str] | 'ALL'): The driver(s) to analyze. Can be a single
                driver name, a list of driver names, or 'ALL' to analyze all drivers.
            num_simulations (int): The number of Monte Carlo simulations to run.
            method ('single-core' | 'multi-core', optional): The processing method to use.
                Defaults to 'multi-core'.
            immediate_plot (bool, optional): If True, generates plots immediately after the simulations.
                Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the results of the Monte Carlo simulations.

        Raises:
            ValueError: If an invalid processing method is specified.
        """
        if drivers_to_analyse == 'ALL':
            drivers_to_analyse = list(self.drivers.keys())
        elif isinstance(drivers_to_analyse, str):
            self._check_driver_validity(drivers_to_analyse)
            drivers_to_analyse = [drivers_to_analyse]
        else:
            for driver in drivers_to_analyse:
                self._check_driver_validity(driver)

        drivers_to_analyse_set = set(drivers_to_analyse)

        if method == 'multi-core':
            pandas_df = self._run_multi_core_monte_carlo(drivers_to_analyse_set, num_simulations)

        elif method == 'single-core':
            pandas_df = self._run_single_core_monte_carlo(drivers_to_analyse_set, num_simulations)

        else:
            raise ValueError("Method must be either 'single-core' or 'multi-core'. "
                             "Multi-core utilizes PySpark (ensure Java is installed for 'multi-core').")
            
        

        # Create a new column with the list version and convert the original to string
        pandas_df['compounds_used_list'] = pandas_df['compounds_used']
        pandas_df['compounds_used'] = pandas_df['compounds_used'].astype(str)

        # copying it so it doesnt change the original
        plot_df = copy.deepcopy(pandas_df)

        if immediate_plot:
            self.plot_all(plot_df)

        return pandas_df

    @time_simulation(message='Simulations complete. Execution time:')
    def _run_single_core_monte_carlo(self, drivers_to_analyse: set, num_simulations: int):
        """
        Run Monte Carlo simulations sequentially on a single core.

        Args:
            drivers_to_analyse (set): Name of the drivers to analyze.
            num_simulations (int): Number of simulations to run.

        Returns:
            pandas.DataFrame: Results of Monte Carlo simulations.
        """

        fields = self._generate_field_names()
        all_results = []
        for _ in tqdm(range(num_simulations)):
            drivers_results = self._mc_simulation(drivers_to_analyse)
            for driver_result in drivers_results:
                driver_result_dict = {field: getattr(driver_result, field) for field in fields}
                all_results.append(driver_result_dict)

        pandas_df = pd.DataFrame(all_results)

        return pandas_df

    @time_simulation(message='Simulations complete. Execution time:')
    def _run_multi_core_monte_carlo(self, drivers_to_analyse: set, num_simulations: int, ):
        """
        Run Monte Carlo simulations on all the cores in parallel using PySpark.

        Args:
            drivers_to_analyse (str): Name of the driver to analyze.
            num_simulations (int): Number of simulations to run.

        Returns:
            pandas.DataFrame: Results of Monte Carlo simulations from distributed processing.
        """
        print(f"Initiating distributed Monte Carlo simulation for {drivers_to_analyse} for {num_simulations} runs.")
        print("Setting up PySpark session.")
        # Turning on Arrow-based columnar data transfers
        spark = SparkSession.builder \
            .appName("Racing Strategy Analysis").config("spark.ui.enabled", "true") \
            .config("spark.ui.port", "4053") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

        # Explicitly defining the schema for the simulation results of monte carlo
        schema = self._generate_schema()

        print("Preparing distributed Monte Carlo simulations.")
        # Create an RDD to represent each simulation task
        simulations_rdd = spark.sparkContext.parallelize([0] * num_simulations)

        simulation_results_rdd = simulations_rdd.flatMap(lambda _: self._mc_simulation(drivers_to_analyse))
        # Convert the RDD of Row objects directly into a DataFrame using the defined schema
        df = spark.createDataFrame(simulation_results_rdd, schema=schema)

        print("Running simulations...")

        pandas_df = df.toPandas()

        return pandas_df

    def plot_all(self, pandas_df: pd.DataFrame):
        """Generate all plots for the Monte Carlo simulation results.

        This method generates a comprehensive set of plots to visualize the results of the
        Monte Carlo simulations. The plots include:
        - Finishing position frequency for each driver
        - Race strategy ranking
        - Race strategy performance
        - Race strategy position distribution
        - Parallel coordinate plot of simulation results
        - Traffic heatmap
        - Traffic boxplot
        - 3D scatter plot of simulation results
        - Basic 3D scatter plot of simulation results
        - Driver position dominance

        Args:
            pandas_df (pd.DataFrame): A pandas DataFrame containing the Monte Carlo simulation results.
        """

        plot_drivers_finishing_position_frequency(pandas_df)
        plot_race_strategy_ranking(pandas_df)
        self.plot_race_strategy_performance(pandas_df)
        plot_race_strategy_position_distribution(pandas_df)
        self.plot_parallel_coordinate_plot(pandas_df)
        plot_traffic_heatmap(pandas_df)
        plot_traffic_boxplot(pandas_df)
        self.plot_three_dimensional_scatter_plot(pandas_df)
        self.plot_three_dimensional_scatter_plot_basic(pandas_df)
        plot_drivers_position_dominance(pandas_df)

    def plot_race_strategy_performance(self, pandas_df: pd.DataFrame):
        """Plot race strategy performance.

        This method generates an interactive plot to visualize the performance of different race
        strategies for each driver. The plot shows the relationship between the first pit stop lap
        and the mean finishing position for each strategy. The strategies are fitted with a
        quadratic curve to highlight the trend.

        Args:
            pandas_df (pd.DataFrame): A pandas DataFrame containing the Monte Carlo simulation results.
        """

        drivers = pandas_df['driver'].unique()
        fig = go.Figure()
        colors = plotly.colors.qualitative.Plotly

        for driver in drivers:
            driver_df = pandas_df[pandas_df['driver'] == driver]
            fields = self._generate_field_names()

            # All drivers have the same index for when their pit stops start
            first_pit_lap = fields[self._pit_stops_start_idx]

            for i, strategy in enumerate(driver_df['compounds_used'].unique()):
                subset = driver_df[driver_df['compounds_used'] == strategy]
                x = subset[first_pit_lap].values
                y = subset['position'].values  # Changed from 'points' to 'position'

                coefficients = np.polyfit(x, y, 2)
                polynomial = np.poly1d(coefficients)

                x_fit = np.linspace(x.min(), x.max(), 1000)
                y_fit = polynomial(x_fit)

                fig.add_trace(
                    go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f"{strategy} ({driver})",
                               line=dict(color=colors[i % len(colors)]),
                               visible=(driver == drivers[0]))
                )

        dropdown_buttons = [dict(
            method='update',
            label=driver,
            args=[{'visible': [driver in trace.name for trace in fig.data]},
                  {
                      'title': f'Pre-Event Race Strategy Evaluations for {driver} (Quadratic Fit)'
                               f' ({count_driver_simulations(pandas_df, driver):,} Simulations)'}]
        ) for driver in drivers]

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=dropdown_buttons,
                x=1.0,
                y=1.1,
                xanchor='right',
                yanchor='top',
            )],
            title_text=f"Pre-Event Race Strategy Evaluations for {drivers[0]} (Quadratic Fit)"
                       f" ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
            xaxis_title='Pit Lap (first stop)',
            yaxis_title='Mean Position',  # Changed from 'Mean Points' to 'Mean Position'
            yaxis=dict(autorange="reversed"),  # Reverse y-axis so lower positions (better) are at the top
            legend_title='Race Strategy',
            legend=dict(x=1.05, y=1, orientation='v', font=dict(size=22)),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        fig.show()

    def plot_parallel_coordinate_plot(self, pandas_df: pd.DataFrame):
        """Plot a parallel coordinate plot of simulation results.

        This method generates an interactive parallel coordinate plot to visualize the Monte Carlo
        simulation results for each driver. The plot shows the relationship between pit stop laps
        and the finishing position. Each driver's strategies are represented as a separate line,
        with colors indicating the finishing position.

        Args:
            pandas_df (pd.DataFrame): A pandas DataFrame containing the Monte Carlo simulation results.
        """

        drivers = pandas_df['driver'].unique()
        fig = go.Figure()

        position_min = pandas_df['position'].min()
        position_max = pandas_df['position'].max()

        custom_colors = ['purple', 'pink', '#32CD32', '#FFFF00', '#DC0000']

        for driver in drivers:
            driver_df = pandas_df[pandas_df['driver'] == driver]
            fields = self._generate_field_names()
            driver_pit_stops_start_idx, driver_pit_stop_end_idx = self._drivers_pit_stop_indices[driver]
            driver_pit_lap_fields = fields[driver_pit_stops_start_idx:driver_pit_stop_end_idx]

            dimensions = [
                *[dict(label=field.replace('_', ' ').title(),
                       values=driver_df[field]) for field in driver_pit_lap_fields],
                dict(label='Position', values=driver_df['position'],
                     range=[driver_df['position'].min(), driver_df['position'].max()])
            ]

            fig.add_trace(
                go.Parcoords(
                    line=dict(color=driver_df['position'],
                              colorscale=custom_colors,
                              showscale=True,
                              colorbar=dict(title="Position", x=1.05, y=0.5),
                              cmin=position_min,
                              cmax=position_max,

                              ),
                    dimensions=dimensions,
                    name=driver,
                    visible=(driver == drivers[0])
                )
            )

        dropdown_buttons = [dict(
            method='update',
            label=driver,
            args=[{'visible': [driver == trace.name for trace in fig.data]},
                  {
                      'title': f"Parallel Coordinate Plot of Race Strategy Simulation Results for {driver} "
                               f"({count_driver_simulations(pandas_df, driver):,} Simulations)"}]
        ) for driver in drivers]

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=dropdown_buttons,
                x=1.0,
                y=1.1,
                xanchor='right',
                yanchor='top'
            )],
            title_text=f"Parallel Coordinate Plot of Race Strategy Simulation Results for {drivers[0]}"
                       f" ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
            showlegend=False,
            margin=dict(r=150)
        )

        fig.show()

    def plot_three_dimensional_scatter_plot(self, pandas_df: pd.DataFrame):
        """Plot a 3D scatter plot of simulation results.

       This method generates an interactive 3D scatter plot to visualize the Monte Carlo simulation
       results for each driver. The plot shows the relationship between the first, second, and third
       pit stop laps, with the color of each point indicating the finishing position. The pit stop
       laps are jittered to improve visibility.

       Args:
           pandas_df (pd.DataFrame): A pandas DataFrame containing the Monte Carlo simulation results.
       """
        drivers = pandas_df['driver'].unique()
        fig = go.Figure()

        position_min = pandas_df['position'].min()
        position_max = pandas_df['position'].max()

        custom_colors = ['purple', 'pink', '#32CD32', '#FFFF00', '#DC0000']

        jitter = (0.1, 0.3)
        lower_bound, upper_bound = jitter

        for driver in drivers:
            driver_df = pandas_df[pandas_df['driver'] == driver].copy()

            fields = self._generate_field_names()
            driver_pit_stops_start_idx, driver_pit_stop_end_idx = self._drivers_pit_stop_indices[driver]
            driver_pit_lap_fields = fields[driver_pit_stops_start_idx:driver_pit_stop_end_idx]

            jittered_fields = [f'{field}_jittered' for field in driver_pit_lap_fields]
            driver_df[jittered_fields] = driver_df[driver_pit_lap_fields].apply(
                lambda x: x + np.random.uniform(lower_bound, upper_bound, x.shape))

            hover_text = [
                f"Driver: {driver}<br>" +
                f"Strategy: {row['compounds_used']}<br>" +
                f"Points: {row['points']}<br>" +
                f"Position: {row['position']}<br>" +
                f"Laps Behind Traffic: {row['laps_behind_traffic']}<br>" +
                f"First Pit Stop: {row[jittered_fields[0]]:.2f}<br>" +
                (f"Second Pit Stop: {row[jittered_fields[1]]:.2f}<br>" if len(jittered_fields) > 1 else "") +
                (f"Third Pit Stop: {row[jittered_fields[2]]:.2f}<br>" if len(jittered_fields) > 2 else "")
                for _, row in driver_df.iterrows()
            ]

            x = driver_df[jittered_fields[0]]
            y = driver_df[jittered_fields[1]] if len(jittered_fields) > 1 else [0] * len(x)
            z = driver_df[jittered_fields[2]] if len(jittered_fields) > 2 else [0] * len(x)

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=driver_df['position'],
                        colorscale=custom_colors,
                        opacity=0.8,
                        colorbar=dict(title="Positions Achieved"),
                        cmin=position_min,
                        cmax=position_max,
                        showscale=True
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=driver,
                    visible=(driver == drivers[0])
                )
            )

        dropdown_buttons = [dict(
            method='update',
            label=driver,
            args=[{'visible': [driver == trace.name for trace in fig.data]},
                  {
                      'title': f"3D Scatter Plot of Race Strategy Simulation Results (Jitter range of {jitter})"
                               f" for {driver} ({count_driver_simulations(pandas_df, driver):,} Simulations)"}]
        ) for driver in drivers]

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=dropdown_buttons,
                x=1.0,
                y=1.1,
                xanchor='right',
                yanchor='top'
            )],
            title_text=f"3D Scatter Plot of Race Strategy Simulation Results (Jitter range of {jitter})"
                       f" for {drivers[0]} ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
            scene=dict(
                xaxis_title='First Pit Stop (jittered)',
                yaxis_title='Second Pit Stop (jittered)',
                zaxis_title='Third Pit Stop (jittered)',
            ),
            showlegend=False
        )

        fig.show()

    def plot_three_dimensional_scatter_plot_basic(self, pandas_df: pd.DataFrame):
        """Plot a basic 3D scatter plot of simulation results.

        This method generates a basic interactive 3D scatter plot to visualize the Monte Carlo
        simulation results for each driver. The plot shows the relationship between the first,
        second, and third pit stop laps, with the color of each point indicating the finishing
        position.

        Args:
            pandas_df (pd.DataFrame): A pandas DataFrame containing the Monte Carlo simulation results.
        """
        drivers = pandas_df['driver'].unique()
        fig = go.Figure()
        position_min = pandas_df['position'].min()
        position_max = pandas_df['position'].max()

        custom_colors = ['purple', 'pink', '#32CD32', '#FFFF00', '#DC0000']

        for driver in drivers:
            driver_df = pandas_df[pandas_df['driver'] == driver]
            fields = self._generate_field_names()
            driver_pit_stops_start_idx, driver_pit_stop_end_idx = self._drivers_pit_stop_indices[driver]
            driver_pit_lap_fields = fields[driver_pit_stops_start_idx:driver_pit_stop_end_idx]

            hover_text = [
                f"Driver: {driver}<br>" +
                f"Strategy: {row['compounds_used']}<br>" +
                f"Points: {row['points']}<br>" +
                f"Position: {row['position']}<br>" +
                f"Laps Behind Traffic: {row['laps_behind_traffic']}<br>" +
                f"First Pit Stop: {row[driver_pit_lap_fields[0]]}<br>" +
                (f"Second Pit Stop: {row[driver_pit_lap_fields[1]]}<br>" if len(driver_pit_lap_fields) > 1 else "") +
                (f"Third Pit Stop: {row[driver_pit_lap_fields[2]]}<br>" if len(driver_pit_lap_fields) > 2 else "")
                for _, row in driver_df.iterrows()
            ]
            x = driver_df[driver_pit_lap_fields[0]]
            y = driver_df[driver_pit_lap_fields[1]] if len(driver_pit_lap_fields) > 1 else [0] * len(x)
            z = driver_df[driver_pit_lap_fields[2]] if len(driver_pit_lap_fields) > 2 else [0] * len(x)

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=driver_df['position'],
                        colorscale=custom_colors,
                        opacity=0.8,
                        colorbar=dict(title="Positions Achieved",  # Format tic
                                      ),
                        cmin=position_min,
                        cmax=position_max,
                        showscale=True,
                        # reversescale=True
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=driver,
                    visible=(driver == drivers[0])
                )
            )

        dropdown_buttons = [dict(
            method='update',
            label=driver,
            args=[{'visible': [driver == trace.name for trace in fig.data]},
                  {
                      'title': f"3D Scatter Plot of Race Strategy Simulation Results (Basic) for {driver}"
                               f" ({count_driver_simulations(pandas_df, driver):,} Simulations)"}]
        ) for driver in drivers]

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=dropdown_buttons,
                x=1.0,
                y=1.1,
                xanchor='right',
                yanchor='top'
            )],
            title_text=f'3D Scatter Plot of Race Strategy Simulation Results (Basic) for {drivers[0]}'
                       f' ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)',
            scene=dict(
                xaxis_title='First Pit Stop',
                yaxis_title='Second Pit Stop',
                zaxis_title='Third Pit Stop'
            )
        )

        fig.show()


if __name__ == '__main__':
    pass

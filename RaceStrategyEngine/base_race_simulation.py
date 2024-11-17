"""
This module defines the BaseRaceSimulation class, which serves as the foundation for race simulation logic.

It encapsulates core functionality for simulating a race, including:
- Lap-by-lap race progression
- Pit stop execution and timing
- Overtaking opportunities and outcomes

This class is designed to be inherited from, not instantiated directly. It provides the fundamental
structure and methods that more specialized race simulation classes can build upon.

Note: This class uses ABC (Abstract Base Class) to signify its role as a base class. While not
strictly enforced, it serves as a hint that this class is not meant to be instantiated for actual use.
The child RaceStrategyEngine class should be used for actual simulation purposes. However, for ease of
flexibility and understanding of simulation logic, direct use is not prevented, allowing the class
to be used to learn how the simulation fundamentally works.

Example:
    # Create an instance of BaseRaceSimulation
    test_race = BaseRaceSimulation(drivers, race_state, race_config)

    # Call the private method just once to run the simulation
    test_race._simulation()

    # Display the result
    test_race.display()

    # Note: Unlike RaceStrategyEngine, BaseRaceSimulation doesn't provide functionality
    # to manage or clear the state of a previous simulation. It's intended for
    # single-run exploration of the core simulation logic.
"""

from abc import ABC
import copy
import random
import numpy as np
import pandas as pd
from .combinatorics import StrategyCombinatorics
from .driver import Driver
from .race_configuration import RaceConfiguration
from .tyre_model import TyreModel

class BaseRaceSimulation(ABC):
    """
    Simulates a race, managing driver performance, race strategies, and race progression.

    This class encapsulates the core logic for simulating a race, including:
    - Lap-by-lap race progression
    - Pit stop execution and timing
    - Overtaking opportunities and outcomes
    - Dynamic strategy adjustments
    - Integration of live race data

    Attributes:
        drivers (dict): Dictionary of Driver objects participating in the race.
        _race_state (dict): Current state of the race, including driver positions and times.
        driver_strategies (dict): Current race strategies for each driver.
        _sim_race_state (dict): Simulated race state, updated during simulation runs.
        _accumulated_race_time (dict): Cumulative race times for each driver.

    The simulation can be run in various modes, including full race simulations
    and live race updates. It provides methods for viewing predicted results
    analyzing race outcomes, and exporting detailed results.
    """

    def __init__(self, drivers: dict[str, Driver], starting_race_grid: dict, race_configuration: RaceConfiguration):
        """
        Initialize the RaceStrategyEngine with drivers, starting grid, and race configuration.

        Args:
            drivers (dict): Dictionary of Driver objects participating in the race.
            starting_race_grid (dict): Initial race grid positions and data.
            race_configuration (RaceConfiguration): Race configuration parameters.

        Raises:
            TypeError: If race_configuration is not an instance of RaceConfig.
        """

        self.drivers = drivers
        self._race_state = starting_race_grid
        self._starting_grid = copy.deepcopy(starting_race_grid)
        self._check_grid_validity()  # checking validity of the inputs
        self.driver_strategies = self._get_starting_strategies()
        
        self.drivers_pit_and_compound_info = None

        # race parameters, essentially track characteristics
        if isinstance(race_configuration, RaceConfiguration):
            self.race_config = race_configuration
        else:
            raise TypeError("race_configuration must be a RaceConfiguration instance")

        # lambdified tyre models from sympy to numpy to speed up calculations
        self._lambdified_drivers_tyre_models = self._get_lambdified_drivers_tyre_models()

        # For custom simulation purposes the tyre models are provided too
        self.drivers_tyre_models = self._get_tyre_models()

        # Lap the race starts from
        self._start_lap = 1

        # Stores the race state for each driver throughout the simulation
        self._sim_race_state = None

        # A line that will be used to make the distinction between prediction and what actually happened with live data
        self._virtual_line = None

        # The following is mainly for plotting and tracking purposes and for the incorporation of external live data
        self._driver_laps_behind_traffic = {driver: 0 for driver in self._race_state}
        #  store the cumulative sum lap times within the _simulation method and will be reset at the start of each run
        self._sim_accumulated_race_time = {f"{driver}": [] for driver in self._race_state}
        # will store the cumulative sum laptimes (for gapper plot) for live data purpose
        self.race_history = pd.DataFrame({driver: [] for driver in self._race_state})

    def _check_grid_validity(self):
        """
        Check if the drivers dictionary matches the race state.

        Raises:
            ValueError: If there is a mismatch between drivers and race state.
        """
        drivers_keys = list(self.drivers.keys())
        race_state_keys = list(self._race_state.keys())

        # checking if they are the same length
        if len(drivers_keys) != len(race_state_keys):
            raise ValueError(
                f'There is a mismatch with drivers and state. Amount of drivers in drivers dict: {len(drivers_keys)}, '
                f'Amount of drivers in race state: {len(race_state_keys)}')

        # check if there are mismatches of drivers entered
        if any(driver not in self._race_state for driver in self.drivers):
            missing = [driver for driver in drivers_keys if driver not in self._race_state]
            if missing:
                if len(race_state_keys) > 20:
                    keys_display = ', '.join(race_state_keys[:20]) + ', ... (and others)'
                else:
                    keys_display = ', '.join(race_state_keys)
                raise ValueError(
                    f'Driver(s) {", ".join(missing)} not recognized in the _race_state. Available: {keys_display}')

        self._validate_positions_in_grid()  # finally just check if positions are valid

    def _validate_positions_in_grid(self):
        """
        Check if no driver has an invalid position.

        Raises:
            ValueError: If any driver has an invalid position or if there are duplicate positions.
        """
        seen_positions = set()  # To track positions already encountered
        error_messages = []  # List to collect all error messages before raising them

        for driver, details in self._race_state.items():
            position = details['position']

            # Check if the position is an integer and greater than 0
            if not isinstance(position, int) or position <= 0:
                error_messages.append(
                    f"Invalid position for {driver}: Positions must be positive integers. Found: {position}")

            # Check for duplicate positions
            if position in seen_positions:
                error_messages.append(
                    f"Duplicate position detected for {driver}: Position {position} is already assigned to more than "
                    f"one driver. ")
            else:
                seen_positions.add(position)

        # Check if there were any errors and raise them all at once
        if error_messages:
            raise ValueError("Errors found in race positions:\n" + "\n".join(error_messages))

    def _check_driver_validity(self, driver: str):
        """
        Check if a driver is valid in the simulation parameters.

        Args:
            driver (str): The name of the driver to check.

        Raises:
            KeyError: If the driver is not recognized in the race state.
        """
        if driver not in self._race_state:
            keys_list = list(self._race_state.keys())
            if len(keys_list) > 20:
                keys_display = ', '.join(keys_list[:20]) + ', ... (and others)'
            else:
                keys_display = ', '.join(keys_list)
            raise KeyError(f'{driver} is not recognized. Available: [{keys_display}]')

    def _get_tyre_models(self) -> dict[str, dict[str, TyreModel]]:
        """
        Retrieve the current tyre models for all drivers in the race simulation.

        Returns:
            dict: A dictionary where keys are driver names and values are their respective tyre models.
        """

        return {name: driver_object.tyre_models for name, driver_object in self.drivers.items()}

    def _get_lambdified_drivers_tyre_models(self) -> dict[str, callable]:
        """
        Fetch lambdified tyre models for all drivers and place them in a dictionary.

        Returns:
            (dict): A dictionary mapping driver names to their lambdified tyre models.

        Note:
            Reducing computational delays by using pre-lambdified expressions.
        """
        return {name: driver_object.lambdified_tyre_models for name, driver_object in self.drivers.items()}

    def _get_starting_strategies(self) -> dict:
        """
        Gets the starting strategies selected by each driver.

        Returns:
            (dict): A dictionary mapping driver names to their initial strategy settings.
        """
        return {name: driver_object.selected_strategy for name, driver_object in self.drivers.items()}

    def _reset_accumulated_race_times(self) -> None:
        """
        Clear the accumulated race times for all drivers.

        Notes:
            The resetting is so that each simulation can start from a clean slate,
            and the remnants of the previous one is discarded, allowing the
            same instance to run multiple run consecutive simulations
        """
        for driver in self._sim_accumulated_race_time:
            self._sim_accumulated_race_time[driver].clear()

    def _reset_traffic_counter(self) -> None:
        """
        Reset the traffic counter for each driver to zero.

        Notes:
            The resetting is so that each simulation can start from a clean slate,
            and the remnants of the previous one is discarded, allowing the
            same instance to run multiple run consecutive simulations
        """
        for driver in self._driver_laps_behind_traffic:
            self._driver_laps_behind_traffic[driver] = 0

    def _reset_race_history(self) -> None:
        """
        Reset the race history dataframe.

        Notes:
            The resetting is so that each simulation can start from a clean slate,
            and the remnants of the previous one is discarded, allowing the
            same instance to run multiple run consecutive simulations
        """
        # Need to find smarter way to do this! I cant keep making new dataframes.
        self.race_history = pd.DataFrame({driver: [] for driver in self._race_state})

    def _check_overtake(self, driver: str, current_lap: int) -> None:
        """Simulate and process overtaking opportunities for a given driver at the end of each lap.

        This method evaluates the potential for a driver to overtake cars ahead, considering:
        - Current race positions
        - Time gaps between drivers
        - DRS availability and effect
        - Probabilistic overtaking chances based on time difference
        - Penalties for successful and failed overtake attempts

        The method updates the race state, including driver positions and lap times, based on
        the outcomes of overtaking attempts. It also tracks laps spent behind traffic for each driver.

        Args:
            driver (str): Name of the driver attempting to overtake.
            current_lap (int): The current lap number in the race.

        Notes:
            - This method modifies the `_sim_race_state` and `driver_laps_behind_traffic` attributes.
            - Overtaking is processed from the driver's current position up to the race leader.
            - The method breaks the loop after a failed overtake attempt, assuming the driver
            cannot overtake cars further ahead in the same lap.
        """
        current_position = self._sim_race_state[driver]["position"]
        drs_boost = self.race_config.drs_boost

        if current_position > 1:  # No point checking if first can overtake anyone as he is in the lead

            # Check if there's a car ahead to potentially overtake
            for position in range(current_position - 1, 0, -1):

                # Find driver ahead in the current position
                driver_ahead = next((driver for driver, driver_information in self._sim_race_state.items() if
                                     driver_information["position"] == position), None)

                if driver_ahead is not None:

                    # the race time of driver currently behind
                    driver_total_time = self._sim_race_state[driver]["total_time"]
                    # the race time of the driver currently ahead of him
                    driver_ahead_total_time = self._sim_race_state[driver_ahead]["total_time"]

                    # Explicit check: that he is behind just for clarity
                    driver_is_trailing = driver_total_time > driver_ahead_total_time
                    # The driver behind has greater race time currently, so I subtract from him
                    # to find the gap
                    time_gap = driver_total_time - driver_ahead_total_time

                    # DRS checks
                    within_drs_range = time_gap < self.race_config.delta_for_drs_activation
                    drs_is_active = current_lap >= self.race_config.drs_activation_lap
                    # Checking if driver can gain DRS
                    if driver_is_trailing and within_drs_range and drs_is_active:
                        # Gain the benefit of DRS
                        self._sim_race_state[driver]["total_time"] -= drs_boost

                    # Check overtaking conditions - if the driver behind would 'preliminary' overtake the driver ahead:
                    # Would the DRS have helped the driver behind end up ahead if he was able to phase
                    # through driver in front of him like a 'ghost'
                    if self._sim_race_state[driver]["total_time"] < self._sim_race_state[driver_ahead]["total_time"]:
                        #  From the condition above driver would have lower time than the driver ahead now
                        #  So for now he is 'preliminary' ahead if he was able to glide past

                        # Time advantage the driver behind 'would have'
                        time_gap = self._sim_race_state[driver_ahead]["total_time"] - self._sim_race_state[driver][
                            "total_time"]

                        # Note: The overtake_chance can be set as threshold rather than
                        # probabilistic in the RaceConfiguration. Checking what would the chance of an
                        # overtake be considering you can't phase through a car.
                        # Determining what the time advantage (time_gap) above puts the probability of success

                        # Overtake metric used to calculate probability of success
                        overtake_chance = self.race_config.overtake_chances(time_gap)
                        # Extra Note if a threshold was used in RaceConfiguration:
                        # You would need a certain amount of advantage to pull of a successful overtake, and
                        # if the time gap was above this the overtake_chance would be either 1 ie 100% or 0 ie 0%
                        # binary in choice.

                        # Now checking if the time advantage is sufficient for a successful overtake
                        # This uses a random factor to simulate real-world variability in overtaking situations
                        if random.random() < overtake_chance:
                            # Overtake successful. Swap positions
                            self._sim_race_state[driver]["position"] -= 1
                            self._sim_race_state[driver_ahead]["position"] += 1

                            self._sim_race_state[driver]["total_time"] += self.race_config.time_lost_performing_overtake

                            self._sim_race_state[driver_ahead][
                                "total_time"] += self.race_config.time_lost_due_to_being_overtaken

                        else:  # Unsuccessful overtake attempt
                            # Since the simulation is lap by lap and overtake attempt is only allowed if faster this
                            # is simple heuristic decision
                            self._driver_laps_behind_traffic[driver] += 1
                            # time lost when the driver fails to overtake the car ahead, it can randomly range from
                            # the min to max
                            failed_overtake_penalty = random.uniform(
                                self.race_config.min_time_lost_due_to_failed_overtake_attempt,
                                self.race_config.max_time_lost_due_to_failed_overtake_attempt)

                            # The driver adopts the time of driver he was trying to overtake,
                            # with the additional penalty applied for failing to overtake,
                            # because battling for positions causes you to lose time
                            self._sim_race_state[driver]["total_time"] = self._sim_race_state[driver_ahead][
                                                                             "total_time"] + failed_overtake_penalty
                            

                            break  # Break out the loop. Failed to overtake the car ahead so incapable of overtaking the
                            # car ahead of it, thus no need to check the rest

                else:  # Move the driver up to position ahead if the driver isn't there anymore
                    self._sim_race_state[driver]["position"] -= 1

    def _update_positions_post_pit_stops(self, pitted_drivers: list) -> None:
        """Adjust race positions after drivers complete their pit stops while maintaining race integrity.

        This method updates the race state to reflect new positions after pit stops,
        ensuring that the integrity of the race order is preserved for accurate
        overtaking simulations. It processes pitted drivers from slowest to fastest based on their race times,
        moving them back in the race order based on their new total times, making
        only the minimum necessary position changes. Whilst leaving other drivers alone for the
        _check_overtake method to handle.

        Args:
            pitted_drivers (list): A list of driver names (str) who have just completed their pit stops.

        Notes:
            - The method modifies the `_sim_race_state` attribute in-place.
            - Drivers not in `pitted_drivers` maintain their relative positions to each other.
            - The `pitted_drivers` list is cleared at the end of the method.
            - This method is designed to work in conjunction with the `_check_overtake`
            method, ensuring that on-track overtakes can be accurately simulated in
            subsequent laps.

        Example:
            If the race state received at the end of a lap is:
            1. Hamilton (90s), 2. Vettel (110s), 3. Leclerc (89.7s)
            And Vettel is in the pitted_drivers list, it will update to:
            1. Hamilton (90s), 2. Leclerc (89.7s), 3. Vettel (110s)

            This allows for potential overtakes to be simulated in following laps based on the new race order and
            time gaps. So that drivers close to each other aren't just pushed ahead of the other, without a
            'battle'/'interaction' whilst they are on track. Pitstop drivers aren't on track, so they can be slotted
            in where their race time (total accumulated lap time) allows.
        """

        # Sort pitted_drivers based on their post-pitstop race time in descending order,
        # Reason is that I found that sort pitted drivers from slowest to fastest to ensure correct positioning
        # This avoids conflicts when multiple drivers pit and maintains race order integrity
        sorted_pitted_drivers = sorted(pitted_drivers,
                                       key=lambda driver: self._sim_race_state[driver]["total_time"], reverse=True)

        for pitted_driver in sorted_pitted_drivers:
            old_position = self._sim_race_state[pitted_driver]["position"]
            positions_to_move = 0

            for driver, driver_data in self._sim_race_state.items():
                # Checking the drivers below the driver who has pitted and also if they have a lower race time if so
                # increment because he should be behind them
                driver_was_behind = driver_data["position"] > old_position
                now_ahead_on_track = driver_data['total_time'] < self._sim_race_state[pitted_driver]['total_time']
                if driver_was_behind and now_ahead_on_track:
                    positions_to_move += 1

            new_position = old_position + positions_to_move
            pitted_driver_has_lost_positions = positions_to_move > 0
            # Update positions for affected drivers
            if pitted_driver_has_lost_positions:
                for driver in self._sim_race_state:
                    # Move the drivers up positions if the pitted driver was initially ahead
                    # of them but now comes out behind them after the pitstop
                    pitted_driver_came_out_behind = (
                            old_position < self._sim_race_state[driver]["position"] <= new_position
                    )
                    if pitted_driver_came_out_behind:
                        self._sim_race_state[driver]["position"] -= 1

            # Move the pitted driver to his correct position
            self._sim_race_state[pitted_driver]["position"] = new_position

        # Clearing the list once completed updating positions for pitted drivers and repositioning surrounding drivers
        pitted_drivers.clear()

    def _compute_lap_time(self, current_lap: int, driver: str, tyre_model_contribution_to_lap_time: float) -> float:
        """
        Calculate the adjusted lap time for a driver based on various race factors.

        This method computes the lap time considering factors such as:
        - Driver consistency (mean deviation and standard deviation)
        - Fuel load and its effect on lap time
        - Special conditions for the first lap (race start penalties)

        Args:
            current_lap (int): The current lap number in the race.
            driver (str): The name of the driver.
            tyre_model_contribution_to_lap_time (float): The initial lap time calculated based on tyre performance.

        Returns:
            float: The adjusted lap time for the driver.

        Notes:
            The first lap includes additional time penalties to account for the standing start
            and the effect of the driver's grid position.
        """

        driver_object = self.drivers[driver]

        # The fuel effect on lap time is calculated based on the current amount of fuel and the
        # race configuration parameter that specifies the effect of each kg of fuel on the lap time.
        current_amount_of_fuel = self.race_config.total_fuel - self.race_config.fuel_consumption_per_lap * current_lap
        fuel_effect = current_amount_of_fuel * self.race_config.fuel_effect_seconds_per_kg

        # The driver_effect / driver's consistency for lap is determined by taking the maximum of a minimum variation
        # and a random value drawn from a normal distribution defined by the driver's mean lap time deviation and
        # standard deviation.
        driver_effect = max(driver_object.min_lap_time_variation,
                            random.gauss(driver_object.lap_time_mean_deviation,
                                         driver_object.lap_time_std_dev))

        if current_lap == 1:  # Consider race start effects on the first lap
            # The grid position time effect is calculated by multiplying the driver's starting position by a
            # penalty factor defined in the race configuration.
            race_start_grid_position_time_effect = (self._race_state[driver]['position'] *
                                                    self.race_config.race_start_grid_position_time_penalty)

            # The stationary time effect is a fixed penalty for the time lost due to starting from a
            # stationary position, as defined in the race configuration.
            race_start_stationary_time_effect = self.race_config.race_start_stationary_time_penalty

            # The total race start effect is the sum of the stationary time effect and the grid position time effect.
            race_start_effect = race_start_stationary_time_effect + race_start_grid_position_time_effect

            # On the first lap, the true lap time is the sum of the tyre model contribution, the race start effect,
            # the driver's consistency, and the fuel effect.
            true_lap_time = tyre_model_contribution_to_lap_time + race_start_effect + driver_effect + fuel_effect

        else:
            # On all other laps, the true lap time is the sum of the tyre model contribution, the driver's consistency,
            # and the fuel effect (i.e., no race start effects).
            true_lap_time = tyre_model_contribution_to_lap_time + driver_effect + fuel_effect

        return true_lap_time

    def _get_drivers_pit_and_compound_info(self) -> dict:
        """Retrieve pit stop and compound information for each driver.

        This method processes the driver strategies to extract the pit stop laps and tyre compound
        information for each driver. It calculates the cumulative laps for each compound and rounds
        the laps on compounds to ensure integer values.

        Returns:
            dict: A dictionary containing pit stop and compound information for each driver, with the

        Notes:
            - The method iterates over the `driver_strategies` attribute to process each driver's strategy.
            - The `smart_round` method is used to round the laps on compounds to ensure they are integers
            and sum up to the total number of laps in the race.
            - The pit stop laps are calculated as the cumulative laps excluding the last stop.
        """
        drivers_lap_info = {}
        for driver in self._race_state:
            driver_object = self.drivers[driver]

            available_tyres = driver_object.available_tyres_constraint
            available_tyres_method = driver_object.available_tyres_method
            driver_name = driver_object.name
            # print(driver, self.driver_strategies[driver])
            compounds, laps_on_compounds = self._extract_strategy_compounds_and_laps(driver, included_race_time=True)

            # Rounding to ensure the pit laps are integers and not floating points from the optimization method
            rounded_laps_on_compounds = StrategyCombinatorics.smart_round(laps_on_compounds, self.race_config.num_laps)

            # Calculate cumulative laps
            cumulative_laps = np.cumsum(rounded_laps_on_compounds)

            # Calculate pit laps
            pit_laps = cumulative_laps[:-1]
            pit_out_laps = set(pit_laps + 1)  # Create set for fast lookup of the out laps

            if available_tyres:
                # Match the compounds used in the strategy to the appropriate number of laps they've been previously
                # used if applicable. By default, it prioritizes selecting the 'newest' (least used) compounds first.
                lap_num_on_compound = StrategyCombinatorics.match_strategy_compounds_and_laps_used(
                    compounds, available_tyres, driver_name, method=available_tyres_method)

            else:  # Default to all the tyres are new
                lap_num_on_compound = [0] * len(compounds)

            drivers_lap_info[driver] = {"compounds": compounds,
                                        "cumulative_laps": cumulative_laps,
                                        'laps_on_compound': rounded_laps_on_compounds,
                                        # List of pit stop laps (for indexing compatibility)
                                        'pit_laps':  pit_laps,
                                        # Note - creating a set of pit stop laps:
                                        # 1. cumulative_laps[:-1] excludes the last element (no pit after final stint)
                                        # 2. + 1 shifts time penalty to occur after completing each stint
                                        # This is for effectively adding the out-lap portion to the correct lap
                                        # The 'in-lap' would be on lap 24, for example, but you cross the line in the
                                        # pit, so the time loss will be seen on the out-lap, i.e., lap 25
                                        # 3. set() for efficient lookup during race simulation 
                                        'pit_out_laps':  pit_out_laps,
                                        # This key is effectively the starting tyre age of all the compounds to
                                        # be used
                                        # TODO change this too to reflect the actual nature of the variable something like intial laps on compound
                                        'lap_num_on_compound': lap_num_on_compound}

        return drivers_lap_info
    # TODO IF OPTIONAL STRtegy is given driver doesnt have to be
    def _extract_strategy_compounds_and_laps(self, driver, optional_strategy = None, included_race_time: bool = True):
        strategy_to_use = optional_strategy if optional_strategy is not None else self.driver_strategies[driver]
        
        if included_race_time:

            # print(f'{driver}: {strategy_to_use}')
            _, *compound_strategy = strategy_to_use
        else:
            compound_strategy = strategy_to_use
        # Process compound strategy, to extract the compounds and respective laps on them
        compounds, laps_on_compounds = zip(*compound_strategy)
        return compounds, laps_on_compounds

    def _simulation(self) -> None:
        """Performs the race simulation logic.

        This method runs the core simulation logic for the race, The simulation is run from the
        starting lap to the total number of laps in the race with the starting grid entered, additionally
        incorporating live data when available through (live_state_updates) method.
        And handling various aspects such as pit stops, tyre strategies, lap time calculation,
        and overtaking.

        The method updates the `_sim_race_state` attribute with the latest race information,
        including lap times, total times, and driver positions. It also maintains the
        `accumulated_race_time` attribute to track the cumulative race times for each driver.

        The simulation follows these key steps:
        1. Initialize the simulation race state and accumulated race time.
        2. Retrieve driver pit stop and compound information.
        3. Iterate through each lap of the race.
        4. For each driver, determine the tyre compound and calculate the lap time.
        5. Handle pit stops and update the race state accordingly.
        6. Process overtaking opportunities from the driver in 2nd to the last position. (first can't overtake anyone)
        7. Update the accumulated race time for each driver.

        Notes:
            - The simulation uses a deepcopy of the `_race_state` to ensure data integrity.
            - The `_start_lap` attribute determines the starting point of the simulation.
            - The `race_params` attribute provides various race-related parameters.
            - The `driver_strategies` attribute contains the pit stop and tyre strategies for each driver.
            - The `lambdified_drivers_tyre_models` attribute holds the lap time calculation models.
            - The `_compute_lap_time` method is used to calculate the lap time for each driver.
            - The `_perform_pitstop` method handles the pit stop logic and updates driver positions.
            - The `_check_overtake` method simulates overtaking opportunities for each driver.

        Side Effects:
            - Updates the `_sim_race_state` attribute with the latest race information.
            - Modifies the `accumulated_race_time` attribute to track cumulative race times.
            - Updates the `_sim_accumulated_race_time` attribute with the final accumulated race times.
        """
        # Using this deepcopy below so the simulation can be run consecutively, returning back to the
        # default if need be. This is mainly done due to the mutability of dictionaries, this allows for clear
        # separation of concerns. The self._race_state will be used as a baseline where the simulation start point is
        # and to avoid changing this The self._sim_race_state builds on it without ever-changing the baseline This
        # avoids added complexity of managing when and where to clear the state for different situations such offline
        # simulations and the incorporating of live data that will be sent to _race_state in live_state_method
        # random.seed(21)
        
        self._sim_race_state = copy.deepcopy(self._race_state)
        # acting as a pointer to instance variable of the same name # to save time just removing the self
        # self.saved_race_state = []
        
        # Always call the function to get the latest changes at the start of the simulation
        self.drivers_pit_and_compound_info = self._get_drivers_pit_and_compound_info()
        # just for ease so i dont have to put self, they point to the same its a dict
        drivers_lap_info = self.drivers_pit_and_compound_info

        pitted_drivers = []
        for current_lap in range(self._start_lap, self.race_config.num_laps + 1):

            for driver in self._sim_race_state:
                # Note!: Mutability use
                # Leveraging the mutability of lists, the
                # changes made within this methods will affect the original object,
                # the pitted_drivers list defined earlier
                # This approach is used for efficiency, avoiding unnecessary copying.
                # Note: This call will modify pitted_drivers (if pitting on that lap)
                driver_lap_time = self._get_driver_lap_time_and_update_pit_status(driver,
                                                                                  current_lap, drivers_lap_info,
                                                                                  pitted_drivers)

                self._sim_race_state[driver]['total_time'] += driver_lap_time

            if pitted_drivers:
                self._update_positions_post_pit_stops(pitted_drivers)

            # Sort drivers by position, from first to last
            sorted_drivers_position = sorted(self._sim_race_state,
                                             key=lambda driver: self._sim_race_state[driver]['position'])

            # Process overtaking opportunities from lead to last position
            for driver in sorted_drivers_position:
                # This top-down approach ensures ive done is mainly for the following
                # Realistic race flow: Lead cars have first overtaking opportunity, and ensures sequential integrity
                # Complementary to bottom-up pitstop logic and helps maintains race dynamics accuracy
                self._check_overtake(driver, current_lap)
            # self.saved_race_state.append(copy.deepcopy(self._sim_race_state))
            for driver in self._sim_race_state:
                self._sim_accumulated_race_time[driver].append(self._sim_race_state[driver]['total_time'])
    
    def _get_driver_lap_time_and_update_pit_status(self, driver: str, current_lap: int, drivers_lap_info: dict,
                                                   pitted_drivers: list) -> float:
        """Calculate the lap time for a driver and apply pit stop effects if it's the driver's out lap.

        This method determines the driver's lap time based on their current tyre compound
        and lap number. It also checks if the current lap is the driver's out lap after a pit stop
        and applies the appropriate time penalty. If it is an out lap, the driver is added to the
        pitted_drivers list.

        Args:
            driver (str): The name of the driver.
            current_lap (int): The current lap number.
            drivers_lap_info (dict): Information about each driver's pit stops and tyre compounds.
            pitted_drivers (list): A list of drivers who are on their out lap after pitting.
                                This list is modified in-place if the current lap is an out lap.

        Returns:
            float: The calculated lap time for the driver, including any pit stop effects.

        Side effects:
            - Modifies the pitted_drivers list if the current lap is the driver's out lap.
        """
        driver_object = self.drivers[driver]

        # Default values for the start of the race
        stint = 0  # Starting stint (first stint)
        tyre = drivers_lap_info[driver]["compounds"][stint]
        laps_to_run_compound = drivers_lap_info[driver]['laps_on_compound'][stint]
        stint_end_lap = drivers_lap_info[driver]["cumulative_laps"][stint]
        # Using set for fast lookup of pit laps, optimizing 'in' operation performance
        pit_out_laps = drivers_lap_info[driver]["pit_out_laps"]

        # Determine the current tyre compound and update lap count based on the current lap number
        # stint_end_lap: lap number where the current tyre stint ends
        #
        # This approach uses cumulative lap numbers to determine stint changes and track tyre usage:
        # - Each value in cumulative_laps represents the last lap of a stint
        # - The index of each cumulative_lap corresponds to a tyre compound in the compounds list
        # - lap_num_on_compound tracks the number of laps run on each compound
        #
        # Example:
        # compounds: ['soft', 'medium', 'hard']
        # lap_num_on_compound: [0, 0, 0]  (initially)
        # cumulative_laps: [20, 40, 55]
        # This means:
        #   Stint 1: soft tyres, laps 1-20
        #   Stint 2: medium tyres, laps 21-40
        #   Stint 3: hard tyres, laps 41-55
        #
        # As the race progresses, lap_num_on_compound is updated to reflect actual tyre usage
        # e.g., After lap 25: [20, 5, 0]  (20 laps on soft, 5 on medium, 0 on hard)
        for stint, stint_end_lap in enumerate(drivers_lap_info[driver]["cumulative_laps"]):
            if current_lap <= stint_end_lap:
                tyre = drivers_lap_info[driver]["compounds"][stint]
                laps_to_run_compound = drivers_lap_info[driver]['laps_on_compound'][stint]
                break  # if found no point in checking the rest

        # Calculate proportion of lap time that will come from the appropriate tyre model
        lap_num_on_compound, current_tyre_age = self._get_laps_on_compound(current_lap, driver, drivers_lap_info, stint)

        initial_lap_time_calc = self._lambdified_drivers_tyre_models[driver][f'{tyre}'](lap_num_on_compound)

        # Compute final lap time that will include things such as driver consistency, fuel effect, etc
        lap_time = self._compute_lap_time(current_lap, driver, initial_lap_time_calc)
        # self._sim_race_state
        self._sim_race_state[driver]['current_tyre'] = tyre
        self._sim_race_state[driver]['current_tyre_laps_age'] = current_tyre_age

        if current_lap == stint_end_lap:
            # Add the information about the tyres used in the stint to the race state
            # TODO temporary solution for now later ill enforce the use of used tyres, and simplify the line below
            # self._sim_race_state[driver]['used_tyres'] = self._sim_race_state[driver].get('used_tyres', [])
            used_tyre_stint = tuple((tyre, laps_to_run_compound))

            self._sim_race_state[driver]['used_tyres'].append(used_tyre_stint)

        # Check if this is the driver's out lap after a pit stop and apply the appropriate time penalties for
        # performing a pitstop and coming out the pit lane
        if current_lap in pit_out_laps:
            tyre_change_time = random.gauss(driver_object.mean_tyre_change_time, driver_object.std_dev_tyre_change_time)
            pit_lane_loss = self.race_config.pit_lane_time_loss
            total_pit_loss = tyre_change_time + pit_lane_loss

            lap_time += total_pit_loss

            # Caution: Mutability use
            # This is the mutable list, whatever I add here will be visible in the original object in the _simulation
            pitted_drivers.append(driver)

        return lap_time

    def _get_laps_on_compound(self, current_lap, driver, drivers_lap_info, stint):
        previous_cumulative_lap = drivers_lap_info[driver]["cumulative_laps"][stint - 1] if stint > 0 else 0
        initial_laps_used = drivers_lap_info[driver]['lap_num_on_compound'][stint]

        current_race_tyre_age = current_lap - previous_cumulative_lap

        # lap_num_on_compound is to be the lap number since the last tyre change ie number of laps on that compound
        # Calculate the number of laps on the current compound
        # The subtract by 1 is so that if a tyre is new ie not being used lap on compound is 0 not 1
        lap_num_on_compound = current_race_tyre_age + initial_laps_used - 1
        return int(lap_num_on_compound), int(current_race_tyre_age)

    def display(self) -> None:
        """
        Display the results of the race in a neat format.

        This method prints out the final grid of the race, showing each driver's position,
        a three-letter abbreviation of their name, and their total race time.

        Returns:
            None

        Raises:
            AttributeError: If the simulation hasn't been run

        Side Effects:
            Prints the race results to the console.
        """
        self._check_simulation_run()
        self._check_validity_simulation_result()

        sorted_drivers = sorted(self._sim_race_state, key=lambda d: self._sim_race_state[d]['position'])
        print('*#' * 50)
        print("FINAL GRID".upper())
        print('-' * 50)
        for driver in sorted_drivers:
            ranking = self._sim_race_state[driver]['position']
            print(f'P{ranking}\t{driver[0:3]} \t\t{round(self._sim_race_state[driver]["total_time"], 2)}')
        print('*#' * 50)

    def _check_simulation_run(self, custom_message=None):
        if not self._sim_race_state:
            default_message = ("Simulation state is not initialized. Please run the simulation first by "
                               "utilizing the 'run_simulation()' method.")
            raise AttributeError(custom_message or default_message)

    def _check_validity_simulation_result(self):
        """Validates the simulation results for consistency and completeness.

        This method performs several checks on the simulation results:
        1. Ensures there are no skipped or duplicate positions.
        2. Verifies that all expected positions are present.
        3. Confirms the presence of a winner (position 1).
        4. Checks if all drivers completed the same number of laps.

        Raises:
            ValueError: If any of the following conditions are met:
                - Duplicate positions are found.
                - There are missing or unexpected positions.
                - No winner (position 1) is found.
                - Not all drivers have the same number of lap times, but completed the race.

        Notes:
            The live_states_update method has functionality builtin that can restructure the race simulation
            if a driver retires.
        """

        # Checking that there is integrity in the positions
        num_drivers_finished_race = len(self._sim_race_state)
        positions = [data["position"] for data in self._sim_race_state.values()]
        expected_positions = set(range(1, num_drivers_finished_race + 1))
        actual_positions = set(positions)

        if len(positions) != len(set(positions)):
            raise ValueError("Duplicate positions found in race results")

        if actual_positions != expected_positions:
            missing = expected_positions - actual_positions
            extra = actual_positions - expected_positions
            error_msg = []
            if missing:
                error_msg.append(f"Missing position(s): {missing}")
            if extra:
                error_msg.append(f"Unexpected position(s): {extra}")
            raise ValueError(f"Position mismatch: {'. '.join(error_msg)}")

        if 1 not in positions:
            raise ValueError("No winner found. No driver in the _sim_race_state that had an end position of 1")

        # self._check_tyre_usage()
        # self._check_minimum_pit_stops()

        # Checking if all drivers that completed the race, completed the same amount of laps
        lengths_of_laptimes = {driver: len(laptimes) for driver, laptimes in self._sim_accumulated_race_time.items()}
        unique_lap_counts = set(lengths_of_laptimes.values())
        if len(set(lengths_of_laptimes.values())) != 1:
            max_laps = max(unique_lap_counts)
            mismatched = [f"{driver}: {length}" for driver, length in lengths_of_laptimes.items() if
                          length != max(lengths_of_laptimes.values())]
            raise ValueError(f'Inconsistent amount of lap times detected in race simulation. '
                             f'Expected all drivers that completed {max_laps} laps '
                             f'to have the same amount of lap times.\n'
                             f'Drivers with incorrect amount of lap times, '
                             f'format - (name, laps completed): {", ".join(mismatched)}')

    def _check_tyre_usage(self):
        drivers_pit_stop_info = self.drivers_pit_and_compound_info
        used_tyres_errors = []
        for driver, result in self._sim_race_state.items():
            total_laps = sum(laps for _, laps in result['used_tyres'])
            compounds_used = [compounds for compounds, _ in result['used_tyres']]
            # ensuring it comes back a tuple no matter what
            used_tyres = [(compounds, laps) for compounds, laps in result['used_tyres']]
            
            answer = drivers_pit_stop_info[driver]
            compounds = answer['compounds']
            laps_on_compounds = answer['laps_on_compound']
            strategy_used_tyres = list(zip(compounds, laps_on_compounds))
            
            if total_laps != self.race_config.num_laps:
                
                used_tyres_errors.append(
                    f"{driver}'s used tyres amounted to {total_laps} laps when it should have been"
                    f" {self.race_config.num_laps}. The lap which the simulation was started: {self._start_lap}.\n "
                    f"The used tyres at the end of the simulation that caused the error error"
                    f" - {result['used_tyres']}\n"
                    f"The used tyres when the simulation was started at lap {self._start_lap}:"
                    f"{self._race_state[driver]['used_tyres']}\n"
                    f"The strategy that the simulation was doing for {driver} was: {strategy_used_tyres}\n"
                    f"The full race state received for {driver} at the start of lap {self._start_lap}:"
                    f"\n{self._race_state[driver]}\n"
                    f"Came from {self.driver_strategies[driver]}\n")
                
            if used_tyres != strategy_used_tyres:
                used_tyres_errors.append(
                    f"{driver}'s actual tyre usage doesn't match the strategy the simulation was doing:\n"
                    f"Actual: {used_tyres}\n"
                    f"Strategy: {strategy_used_tyres}\n"
                    f"Came from {self.driver_strategies[driver]}"
                )

            if len(set(compounds_used)) <= 1:
                used_tyres_errors.append(f"{driver} did not use two different compounds: {driver} used - {compounds_used}"
                                         f"Came from {self.driver_strategies[driver]}")

        if used_tyres_errors:
            error_message = "Data Mismatch:\n" + "\n".join(used_tyres_errors)
            raise ValueError(error_message)

    def _check_minimum_pit_stops(self):
        drivers_without_pitstop = []
        for driver, result in self._sim_race_state.items():
            amount_of_stops = len(result['used_tyres']) - 1
            if amount_of_stops < 1:
                drivers_without_pitstop.append(driver)

        if drivers_without_pitstop:
            raise ValueError(
                f"The following drivers did not perform any pit stops: {', '.join(drivers_without_pitstop)}. "
                f"At least one pit stop is required.")

if __name__ == '__main__':

    from simulation_parameters_example import drivers, race_state, race_config
    import time

    test_race = BaseRaceSimulation(drivers, race_state, race_config)

    # Start the timer
    start_time = time.perf_counter()
    # print(test_race.drivers_consistency)
    # print('n',test_race._get_drivers_pit_and_compound_info())
    # print('METHOD 1 RESULTS')
    # print('METHOD 2 RESULTS')
    # test_race.run_simulation()
    # print(test_race._get_drivers_pit_and_compound_info())

    test_race._simulation()

    # print(test_race._get_drivers_pit_and_compound_info()['Verstappen'])
    # print(test_race._sim_race_state['Verstappen'])
    # print(test_race.driver_strategies['Verstappen'])
    test_race.display()
    # print(test_race._sim_race_state)
    # print('end of race')
    # for driver, info in test_race._sim_race_state.items():
    #     print(driver, info)
    # print(test_race._sim_race_state)
    # for drv, strat in test_race.driver_strategies.items():
    #     print(drv, strat)
    # print()
    # print(test_race._race_state)
    # print('-'*100)
    # print(test_race._sim_race_state)
    #
    # for drv, info in test_race._sim_race_state.items():
    #     compounds, laps = zip(*info['used_tyres'])
    #     if sum(laps) != 55:
    #         print(f'Error for {drv}')
    #         raise ValueError('Yes')
    #     else:
    #         print(f'{drv} is correct')
    #         print(drv, info)
    #         print(test_race.driver_strategies[drv])
    #         print('-'* 100)
    # test_race.run_simulation()
    # test_race.display()
    # print(test_race._driver_laps_behind_traffic)
    # test_race.update_drivers_tyre_usage()
    # test_race.plot_race_trace()

    # print('\n\n')
    # print(test_race.saved_laptimes)
    # print('\n')
    # print(test_race.saved_race_state)
    # driver_laptimes = {}
    # for driver, acc_lap_times in test_race._sim_accumulated_race_time.items():
    #     lap_times = np.diff(acc_lap_times, prepend=0)
    #     # print(driver, sum(lap_times), lap_times, )
    #     driver_laptimes[driver] = lap_times.tolist()
    # # print(driver_laptimes)
    # with open('list_of_race_states.json', 'w') as race_state_file, open('driver_laptimes.json', 'w') as lap_times_file:
    #     json.dump(test_race.saved_race_state, race_state_file, indent=4)
    #     json.dump(driver_laptimes, lap_times_file, indent=4,)

    
    
    # 'stops': amount_of_stops,
    # 'strategy': compounds_used,
    # 'pit_laps': pit_laps,
    # 'tyre_usage': tyre_usage,
    # print(test_race._sim_race_state)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Execution time: {duration} seconds.")

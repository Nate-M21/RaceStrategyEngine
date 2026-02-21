from collections import defaultdict
from functools import cache
from itertools import zip_longest
from typing import Literal
from .utility import safe_get_from_list, RaceDataPacket
from .combinatorics import StrategyCombinatorics
from .race_simulation import RaceSimulation
import random
import numpy as np
import pandas as pd
import copy
from datetime import datetime
import zmq
import os

class LiveStrategy(RaceSimulation):

    def __init__(self, drivers, starting_race_grid, race_configuration, lap_variation_range: tuple[int, int] = (-10, 10), address='*', port=5555, simulation_type = "lap_discrete", track_name = "abu_dhabi", time_step=0.1 ):
        super().__init__(drivers, starting_race_grid, race_configuration)

        self._lap_variation_range = lap_variation_range
        # Precompute strategies by stops for each driver
        self.drivers_strategies_by_stops = {
            driver: self.drivers[driver].get_alternate_strategies(dict_mode=True)
            for driver in self.drivers
        }

        self._socket = None
        self._address = address
        self._port = port
        # TODO Add the ability to change the protocol that zmq uses
        self._protocol = 'tcp'

        # TODO This is nasty need to fix this properly
        self._enable_exp_rust_backend_strategy_engine(simulation_type, track_name, time_step)

    def _enable_exp_rust_backend_strategy_engine(self, simulation_type = "lap_discrete", track_name = "abu_dhabi", time_step=0.1):
        super()._enable_exp_rust_backend(simulation_type=simulation_type, track_name=track_name, time_step=time_step)
        from strategy_engine_core import RealTimeStrategy
        self._real_time_strategy = RealTimeStrategy(self.drivers, self.race_config, simulation_type, track_name, time_step)

   


    def live_state_updates(self, live_race_data: RaceDataPacket, gap_delta_method: bool = True) -> None:
        """Update the race simulation with live data and restart from the next lap.

        This method is designed to be called continuously during a live race to update
        the internal state of the simulation based on real-time data. It resets the
        simulation environment to align with the latest race information, allowing for
        dynamic adaptation to ongoing race conditions.

        Args:
            live_race_data:

            gap_delta_method (bool, optional): If True, uses the delta to leader for time calculations. Default is True.

        Raises:
            ValueError: If the next_lap is beyond the total number of laps in the race, or if
                there's insufficient historical data for any driver.

        Notes:
            This method is crucial for maintaining the accuracy of the simulation
            during a live race scenario. It ensures that predictions and strategy
            calculations are based on the most up-to-date race information.
            It defaults to gap as this updated mre frequently in live races
        """
        current_lap = live_race_data.current_lap
        live_race_state = live_race_data.race_state
        live_race_lap_times = live_race_data.laptimes

        if current_lap > self.race_config.num_laps:
            raise ValueError(
                '\nStop the car!\nStop the car! \nWe have a problem \nSorry mate, need to retire, stop the car'
                '\n\nDiagnosis: cannot run the simulation past the total number of laps in the race.'
                f"\nInvalid lap number: {current_lap} is beyond the total number of {self.race_config.num_laps} laps.")

        # Synchronize driver data structures with the current race state
        live_race_lap_times = self._sync_driver_data_structures(live_race_state, live_race_lap_times)
        # Reset the accumulated_time dictionary for each driver
        # reinitializes simulation variables to their default states before each run, ensuring independence
        # for the simulations. This prevents residual state from
        # previous runs from affecting subsequent ones.
        self._reset_race_history()

        laps_to_take = current_lap - 1
        error_messages = []  # List to collect all error messages before raising them
        for driver in live_race_lap_times:
            if len(live_race_lap_times[driver]) < laps_to_take:
                error_messages.append(
                    f'There is a mismatch between the expected amount of lap times ({current_lap - 1}) and the '
                    f'available lap data for {driver}, which contains ({len(live_race_lap_times[driver])}) laps of data.')

        if error_messages:
            raise ValueError(
                f"To simulate from the lap {current_lap} onward, each driver's lap times array must contain at "
                f"least '{current_lap - 1}' entries. Errors found in the lap time arrays:\n" + "\n".join(
                    error_messages))

        # taking all the laps from 1 to the current lap from array, also useful if a big fixed array of lap times is
        # given, and it fills up ie array of size 100 for each driver but only 71 laps in race making an array of
        # cumulative laptimes for each driver for future plotting and detailed results purposes
        external_new_race_history = {}
        for driver, lap_times in live_race_lap_times.items():
            external_new_race_history[driver] = list(np.cumsum(lap_times[:laps_to_take], dtype=float))
        # For futurproofing and ease, I am taking the external live state and just modifying it, instead of creating
        # new one if in the future I choose to track more things they are already there and all I'd be doing is
        # slight tweaks
        external_new_race_state = copy.deepcopy(live_race_state)

        if gap_delta_method:

            leading_driver = ''
            for driver in live_race_state:
                if live_race_state[driver]['position'] == 1:
                    leading_driver = driver
                    break

            leading_driver_race_time = safe_get_from_list(external_new_race_history[leading_driver],
                                                          index=-1, default=0)  # the total time the leader has

            for driver in external_new_race_state:
                # Updating the value of total time based on the gap to the leader for every driver
                external_new_race_state[driver]['total_time'] = leading_driver_race_time + live_race_state[driver][
                    'delta_to_leader']

                # I am just removing for clarity so when I analyse the results it doesn't confuse me
                external_new_race_state[driver].pop('delta_to_leader', None)

        else:

            for driver in external_new_race_state:
                # the last index is there race time, as new history is an array cumulative  times
                external_new_race_state[driver]['total_time'] = safe_get_from_list(external_new_race_history[driver],
                                                                                   index=-1, default=0)
                # I am just removing for clarity so when I analyse the results it doesn't confuse me
                external_new_race_state[driver].pop('delta_to_leader', None)

                # Adding virtual line for the race trace plot to see the distinction between simulated and actual
        self._virtual_line = current_lap + 1
        self._race_history_list = external_new_race_history
        self.race_history = pd.DataFrame(external_new_race_history)
        # Making the live state gotten from externally the basis of what the simulation will start from
        self._race_state = external_new_race_state

        self._start_lap = current_lap

    def _sync_driver_data_structures(self, race_state: dict,
                                     race_history: dict[str, list[float]]) -> dict[str, list[float]]:
        """
        Synchronize internal data structures with the current race state.

        This method ensures that driver-specific data structures (accumulated race time,
        laps behind traffic, and race history) only contain data for drivers currently in the race.
        It removes data for drivers who have dropped out.

        Args:
            race_state (dict): The current race state containing active drivers.
            race_history (dict): Historical lap time data for each driver.

        Returns:
            dict: The potentially modified race history if there have been removals in the _race_state
            or live_race_state

        Notes:
            This method modifies `self._accumulated_race_time`, `self._driver_laps_behind_traffic`,
            and `race_history`.

        """
        current_drivers = set(race_state.keys())
        accumulated_time_drivers = set(self._sim_accumulated_race_time.keys())
        traffic_counter_drivers = set(self._driver_laps_behind_traffic.keys())
        race_history_drivers = set(race_history.keys())

        # combine all unique elements from these sets
        all_tracked_drivers = accumulated_time_drivers | traffic_counter_drivers | race_history_drivers

        if current_drivers != all_tracked_drivers:

            # All the leftover drivers from the set must be
            drivers_to_drop = all_tracked_drivers - current_drivers
            # removed

            for driver in drivers_to_drop:
                self._sim_accumulated_race_time.pop(driver, None)
                self._driver_laps_behind_traffic.pop(driver, None)
                race_history.pop(driver, None)

            # if drivers_to_drop:
            #     print(f"Drivers removed from race: {', '.join(drivers_to_drop)}")

            # I did this for completeness but this would never happen but im leaving it here
            # Uncomment the following block if you need to handle new drivers
            # new_drivers = current_drivers - all_tracked_drivers
            # for driver in new_drivers:
            #     self._accumulated_race_time[driver] = []
            #     self._driver_laps_behind_traffic[driver] = 0
            #     race_history[driver] = []
            # if new_drivers:
            #     print(f"New drivers added to race: {', '.join(new_drivers)}")

        return race_history


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

        combinations = driver_object.strategy_generator.generate_combinations(filter_strategy_by=filter_compounds_by,
                                                                    minimum_stops=minimum_stops)
        if output == 'LIST':
            
            return combinations

        combination_by_stop = defaultdict(list)

        for combination in combinations:
            amount_of_stops = len(combination) - 1
            combination_by_stop[f"{amount_of_stops}_stop"].append(combination)

        return combination_by_stop

    def _fetch_random_valid_combination(self, driver: str, required_compounds: list, minimum_stops: int = 1):
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

            strat = self._adjust_current_and_future_stint_lengths(driver=driver,
                                                                 past_and_current_stints=past_and_current_stints,
                                                                 future_stints=future_stints,
                                                                 used_tyres=used_tyres)
            return strat

        else:  # if there are no future stints there nothing left to, this tyre is going to the end test it
            strat = self._fetch_adaptive_lap_distribution(driver=driver)

            return strat

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

        compounds = self._fetch_random_valid_combination(driver=driver, required_compounds=required_compounds)

        if required_compounds:
            lap_distribution = self._generate_lap_distribution(driver=driver, strategy_compounds=compounds,
                                                               required_compounds=required_compounds,
                                                               required_laps=required_laps,
                                                               used_tyres=used_stints)

            new_strategy = self.drivers[driver].strategy_creator(new_lap_distributions=tuple(lap_distribution), optional_message=f"Time made: {datetime.now().strftime("%H:%M:%S")}")

            return new_strategy

        else:

            return self._select_alternate_strategy(driver)

    def _apply_adaptive_strategies(self):

        for driver in self._race_state:

            new_strategy = self._generate_adaptive_strategy(driver)


            self.driver_strategies[driver] = new_strategy


    def run_adaptive_simulation(self):
        """Execute a adaptive simulation of a race based on live data.

        This method orchestrates an adaptive simulation of a race by iteratively modifying
        each driver's strategy and adjusting it based on a predefined range
        of lap variations and the current race state.
        After all modifications, it triggers the actual race simulation.

        """

        self._apply_adaptive_strategies()
        self.run_simulation()

    def start_engine(self):
        self._setup_zmq_socket()

        starting_simulation_data = self._get_rust_simulation_data()
        starting_lap = 1
        self._real_time_strategy.ingest_new_data(starting_simulation_data, starting_lap)
        self._real_time_strategy.start_strategy_engine()
        start_time = datetime.now().strftime("%H:%M:%S")
        print(f"Started engine at: {start_time}")

    def run_engine(self, live_data, gap_delta_method: bool):
        self.live_state_updates(live_race_data=live_data, gap_delta_method=gap_delta_method)
        self._apply_adaptive_strategies()
        new_simulation_data = self._get_rust_simulation_data()
        current_lap = self._start_lap
        self._real_time_strategy.ingest_new_data(new_simulation_data, current_lap)

    def stop_engine(self):
        self._real_time_strategy.stop_strategy_engine()
        end_time = datetime.now().strftime("%H:%M:%S")
        print(f"Stopped engine at: {end_time}")

    def get_predictions(self,wait_time_in_sec = 2.5, sleep_time_in_secs = 0.05):
        return self._real_time_strategy.get_predictions(wait_time_in_sec, sleep_time_in_secs)
    
    def _setup_zmq_socket(self):
        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)

        try:
            self._socket.bind(f"tcp://{self._address}:{self._port}")
            print(f"ZeroMQ socket bound to tcp://{self._address}:{self._port}")

        except zmq.error.ZMQError as error:
            print(f"Unable to bind ZeroMQ socket. For address {self._address} Tried port {self._port}: {error}")
            os._exit(1)
    
    def send_zmq_message(self, data):
        if self._socket:
            try:
                self._socket.send_json(data)
            except zmq.ZMQError as e:
                raise ConnectionError(f"Failed to send message via ZeroMQ: {e}")
        else:
            raise RuntimeError("ZeroMQ socket is not initialized. Did you call start_strategy_engine()?")


if __name__ == "__main__":
    from simulation_parameters_example import drivers, race_state, race_config
    import time



    test_race = LiveStrategy(drivers, race_state, race_config)
    test_race._enable_exp_rust_backend()

    start_time = time.perf_counter()

    # for k,v in test_race.driver_strategies.items():
    #     print(k,v)

    # test_race.run_adaptive_simulation()
    

    a = test_race._generate_adaptive_strategy("Hamilton")
    print()
    print("*"*100)
    print("fINAL ANSWER")
    print(a)
    # test_race.display()

   
    end_time = time.perf_counter()

    duration = end_time - start_time
    # print(f"Execution time Python (lap discrete): {duration} seconds.")

    
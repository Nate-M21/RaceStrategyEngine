"""The core of this module is the RaceStrategyEngine class, which orchestrates the simulation
process and manages race state. It can be used for pre-race strategy planning and
real-time race analysis.

It provides functionality for:
- Simulating entire races lap-by-lap, including pit stops and overtaking
- Handling dynamic race conditions and strategy adjustments
- Processing and integrating live race data updates
- Generating detailed race results and performance analytics
- Visualizing race progress and strategy effectiveness within a race considering interaction with other drivers

Additionally, there is extra information on how to incorporate your own simulation logic into the framework.
"""

import copy
# ----------------------------------------------------------------------------------------
import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# If you want to see how to incorporate your own simulation logic, uncomment and read the
# __gather_results_from_custom_sim_logic() and again uncomment the import below and the
# __example_custom_simulation_integration() to see how simple it could be
# ----------------------------------------------------------------------------------------
from external_simulation_integration_test.test_values import end_race_result_test, end_race_laptimes_test
# ----------------------------------------------------------------------------------------
from .base_race_simulation import BaseRaceSimulation
from .tyre_model import TyreModel
from .utility import safe_get_from_list, driver_styles_for_plotting, RaceDataPacket


class RaceSimulation(BaseRaceSimulation):
    """Extends BaseRaceSimulation to provide a complete race simulation environment.

    This class adds functionality for:
    - Handling dynamic race conditions and strategy adjustments
    - Processing and integrating live race data updates
    - Generating detailed race results
    - Visualizing race progress and strategy effectiveness

    It's designed for both pre-race strategy planning and real-time race analysis,
    offering tools for race strategists and analysts to simulate and analyze various scenarios.

    Key features:
    - Live state updates integration
    - Dynamic strategy modifications
    - Result visualization and analysis methods
    - Public methods for running simulations and accessing results

    """

    # TODO add parameters to init:  safety_car_active=False, dnf_allowed=False,
    #  safety_car_probability= None (will be float for percentage), drivers_dnf_probability = None (will be a dict)
    @property
    def simulation_starting_lap(self):
        return self._start_lap

    def update_driver_race_strategy(self, driver, lap_distribution):
        driver_object = self.drivers[driver]
        self.driver_strategies[driver] = driver_object.strategy_creator(lap_distribution,
                                                                        optional_message=f'Strategy created on lap:'
                                                                                         f' {self._start_lap}')

    def update_driver_tyre_usage(self, driver: str):
        """
        Update the tyre usage information for a specific driver based on the current race state.

        This method retrieves the used tyres and current tyre information for the specified driver
        from the race state. It then updates the driver's tyre usage data and recalculates all
        strategy-dependent attributes.

        Args:
            driver (str): The name of the driver whose tyre usage information should be updated.

        Raises:
            KeyError: If the specified driver does not exist in the simulation.

        Notes:
            This method should be called whenever there's a change in the tyre usage during the race.
            Typically, right after the live_states_update method has ingested the live race state

        """
        self._check_driver_validity(driver)

        driver_object = self.drivers[driver]

        # used tyres being empty will not raise an error, it just doesn't add extra constraints on the
        # optimization for that particular driver
        used_tyres = self._race_state[driver].get('used_tyres', [])

        # There driver must always have a current tyre, so I'm doing this so the program 'fails' loudly
        current_tyre = self._race_state[driver]['current_tyre']

        driver_object.update_tyre_usage(used_tyres, current_tyre)

        # Recalculate all dependent attributes
        self._update_driver_strategy_dependent_attributes()

    def cull_driver_tyre_compound(self, driver: str, compound: str):
        """
        Remove a specific tyre compound from a driver's strategy considerations.

        This method removes the specified tyre compound from all future strategy calculations
        for the given driver. It's typically used during a live race when a particular tyre
        compound is performing unexpectedly or is no longer a viable option for the driver.

        Args:
            driver (str): The name of the driver whose strategies should be updated.
            compound (str): The tyre compound to be removed from strategy considerations.

        Side Effects:
            - Updates the driver's tyre models and strategy options.
            - Recalculates all strategy-dependent attributes for the driver.
            - Updates the overall race simulation state.

        Notes:
            This action significantly impacts strategy calculations and predictions for the
            specified driver. It should be used extremely carefully, typically in response to real-time
            unexpected below par tyre performance were you are sure that driver won't consider it. Or you
            as a user don't want to consider it for future predictions for your car.

        Example: race_sim.cull_driver_tyre_compound("Hamilton", "soft") This will remove the 'soft' compound from all
        of Hamilton's future strategy considerations and from his tyre models / compound.
        """
        self._check_driver_validity(driver)

        self.drivers[driver].cull_tyre(compound)
        self._update_driver_strategy_dependent_attributes()

    def update_driver_tyre_models(self, driver: str, new_tyre_models: dict[str, TyreModel]):
        """
        Update tyre models for the specified driver and recalculate dependent attributes.

        This method updates the tyre models for the driver specified in the race state.
        After updating, it recalculates all strategy-dependent attributes to ensure consistency.

        Args:
            driver (str): name of the driver
            new_tyre_models (dict): New tyre models data.

        Raises:
            KeyError: If the driver does not exist in the simulation

        """
        self._check_driver_validity(driver)

        self.drivers[driver].update_tyre_models(new_tyre_models)

        # Recalculate all dependent attributes
        self._update_driver_strategy_dependent_attributes()

    def _update_driver_strategy_dependent_attributes(self):
        """
        Recalculate all strategy-dependent attributes for all drivers.

        This method updates the following attributes:
        - driver_strategies: The starting strategies for each driver.
        - lambdified_drivers_tyre_models: Lambdified versions of tyre models for quick evaluation.
        - drivers_tyre_models: Current tyre models for each driver.

        This method should be called after any changes that affect driver strategies or tyre models.
        """

        # Recalculate all dependent attributes
        self.driver_strategies = self._get_starting_strategies()
        self._lambdified_drivers_tyre_models = self._get_lambdified_drivers_tyre_models()
        self.drivers_tyre_models = self._get_tyre_models()

    def _full_course_yellow(self):
        """Safety car and virtual safety car implementation """
        # Going to need to use this method, if I remove a driver from the race
        #self._sync_driver_data_structures()
        raise NotImplementedError

    def reset_to_initial_state(self):
        """Reset the instance state to its initial configuration.

        This method clears any residual data from live state updates and resets the simulation
        to its starting conditions. It should be used when transitioning from live race updates
        back to running standard simulations with the same instance.

        Notes:
        - Use this method carefully, as it resets the instance.
        - It only needs to be called once to reset the state.
        - After calling this method, the instance will use its internal state for simulations,
        ignoring any previously incorporated live data.
        """
        self._race_state = copy.deepcopy(self._starting_grid)
        self._start_lap = 1
        self._sim_race_state = None
        self._sim_accumulated_race_time = None
        self._virtual_line = None
        self._reset_traffic_counter()
        self._reset_accumulated_race_times()
        self._reset_race_history()

    def run_simulation(self) -> None:
        """
        Executes a standard race simulation.

        This method performs the following operations:
        1. Resets the traffic counter for each driver from the previous simulation.
        2. Clears accumulated race times from the previous simulation.
        3. Runs the core simulation logic.

        The resetting is mainly to allow this function to be re-run or continuously called
        within a loop, ensuring each simulation maintains its own integrity.

        The simulation processes the entire race, lap by lap, by simulating the following:
        - Lap time calculations for each driver
        - Pit stop executions
        - Overtaking scenarios
        - Race position updates

        After execution, the simulation results are stored in instance variable self._sim_race_state

        Notes:
            This method should be called before attempting to access or
            visualize race results.
        """
        self._reset_traffic_counter()
        self._reset_accumulated_race_times()
        self._simulation()

    def _get_predicted_winner_cumulative_time(self) -> np.ndarray:
        """
        Calculate the cumulative lap times for the predicted race winner.

        This method identifies the driver in first position and calculates their
        average lap time. It then creates an array of cumulative lap times based
        on this average, representing a consistent pace throughout the race.

        Returns:
            np.ndarray: An array of cumulative lap times for the predicted winner.

        Notes:
            This method assumes the race winner maintains a consistent pace
            equal to their average lap time throughout the entire race.
        """

        winner_cumulative_time = []
        for _, data in self._sim_race_state.items():
            if data["position"] == 1:
                total_time_avg = data["total_time"] / self.race_config.num_laps
                winner_average = np.full(self.race_config.num_laps, total_time_avg)
                winner_cumulative_time = np.cumsum(winner_average)

        return winner_cumulative_time

    def _get_race_result_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame containing race results and winner's cumulative times.

        This method creates a DataFrame that includes:
        - The lap times for all drivers throughout the race
        - A column with the cumulative average lap times of the predicted winner

        Returns:
            pd.DataFrame: A pandas DataFrame containing the race history and winner's cumulative times.

        Raises:
            AttributeError: If the simulation hasn't been run
        """

        # Get the validated data
        race_history, winner_cumulative_time = self._validate_and_get_race_data()

        # Adding the winner's cumulative times to the dataframe
        race_history['WinnerCumulativeTime'] = winner_cumulative_time

        return race_history

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

    def get_result_from_simulation_run(self, format_json: bool = True, json_indent: Optional[int | str] = None) -> \
            str | dict:
        """
        This method gathers and compiles the results from one simulation run of a race.

        It extracts and processes the simulation state to provide a detailed summary for each driver,
        including their position, points, pit strategy, and total race time. The output is formatted
        as a JSON string to facilitate easy data interchange and further processing.

        Args:
            format_json (bool): Whether to return the result as a JSON string (True) or a Python dictionary (False).
            json_indent (int, str, or None): The indentation level for the JSON output. Use None for compact JSON,
            an integer for space indentation, or a string (like "\t") for custom indentation.

        Returns:
            str or dict: A JSON string if format_json is True, otherwise a Python dictionary.

        Raises:
            AttributeError: If the simulation hasn't been run
        """
        self._check_simulation_run()  # checking if the simulation has been run
        self._check_validity_simulation_result()

        result = self._sim_race_state
        starting_grid = self._starting_grid

        # # # EXTRAS ------------- uncomment for use

        # cumulative_laptimes = self._get_race_result_df().to_dict('list') # getting all cumulative laptimes for drivers

        # difference_to_winner = self.result_for_race_trace_plotting(format_json=False)

        # Creating dictionary that will house the important end of race information that will later be converted JSON
        race_result = {}

        for driver in result:
            end_position = result[driver]['position']
            start_position = starting_grid[driver]['position']

            race_time = result[driver]['total_time']
            used_tyres = result[driver]['used_tyres']

            laps_behind_traffic = self._driver_laps_behind_traffic[driver]

            # # # {1st: 25, 2nd: 18, 3rd: 15, 4th: 12, 5th: 10, 6th: 8, 7th: 6, 8th: 4, 9th: 2, 10th: 1}
            # transform the position to f1 points system
            points = self.race_config.points_distribution.get(end_position, 0)

            compounds_used, laps = zip(*used_tyres)

            pit_laps = np.cumsum(laps)[:-1]
            # Ensure Python int type for serialization purposes
            pit_laps = [int(lap) for lap in pit_laps]

            amount_of_stops = len(used_tyres) - 1

            tyre_usage = used_tyres
            # # # EXTRAS --------------------------------

            # # # # driver's cumulative race time ie lap times throughout the simulation
            # driver_cumulative_laptimes = cumulative_laptimes[driver]

            # # # # # driver's lap times throughout the simulation
            # laptimes = np.diff(driver_cumulative_laptimes,prepend=0)
            # # need to convert to standard python type for serialization purposes
            # laptimes = [float(lap_time) for lap_time in laptimes]

            # # # # # driver's difference to winner throughout the simulation
            # driver_diff_to_winner = difference_to_winner[driver]

            # dictionary structure of information each driver will have at the end of a simulation
            driver_race_end_info = {
                'simulation_start_lap': self._start_lap,
                'start_position': start_position,
                'end_position': end_position,
                'points': points,
                'stops': amount_of_stops,
                'strategy': compounds_used,
                'pit_laps': pit_laps,
                'tyre_usage': tyre_usage,
                'race_time': race_time,
                'laps_behind_traffic': laps_behind_traffic
                # # # EXTRAS --------------------------------
                # 'lap_times': laptimes,
                # 'cumulative_lap_times': driver_cumulative_laptimes,
                # 'difference_to_winner': driver_diff_to_winner

            }

            race_result[driver] = driver_race_end_info

        return json.dumps(race_result, indent=json_indent) if format_json else race_result

    def _validate_and_get_race_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Check the validity of the simulation run and ensure it has been run before retrieving results.

        This method verifies that the simulation has been run and returns the race history dataframe
        and cumulative time of the winner for further data processing purposes.

        Returns:
            tuple: A tuple containing the following elements:
                - race_history_df (pandas.DataFrame): The race history dataframe, concatenated with the
                accumulated race time from the simulation.
                - winner_cumulative_time (list): The cumulative time of the predicted winner.

        Raises:
            ValueError: If the simulation has not been run or if there is a data length mismatch between
                the winner's cumulative time and the race history dataframe. This can occur if live state
                updates are being used and the simulation is not starting from the correct 'next' lap.

        Notes:
            - The method uses the `_check_simulation_run` method to ensure the simulation has been run.
            - The race history dataframe is created by concatenating the initial `race_history` attribute
            with the accumulated race time from the simulation (`_sim_accumulated_race_time`).
            - The `_find_predicted_winner` method is used to calculate the winner's cumulative time.
            - If there is a data length mismatch, the method raises a `ValueError` with a detailed error
            message, including instructions for using live state updates correctly.
        """

        self._check_simulation_run()  # first check if the simulation has been run
        self._check_validity_simulation_result()  # then check if the result is valid

        race_history_drivers = set(self.race_history.columns)
        accumulated_time_drivers = set(self._sim_accumulated_race_time.keys())

        if race_history_drivers != accumulated_time_drivers:
            missing_in_history = accumulated_time_drivers - race_history_drivers
            missing_in_accumulated = race_history_drivers - accumulated_time_drivers
            raise ValueError(f"Data mismatch: "
                             f"Drivers missing from race history: {missing_in_history or None}, "
                             f"Drivers missing from accumulated race time: {missing_in_accumulated or None}")

        # Combine initial race history (if any) with simulated race data
        # Initial race history may be empty if no live data is used, or contain values from previous laps
        # Concatenate this with the accumulated race time from the simulation
        race_history_df = pd.concat([self.race_history, pd.DataFrame(self._sim_accumulated_race_time)],
                                    ignore_index=True)

        # Calculate the winner's cumulative time using the existing method
        winner_cumulative_time = self._get_predicted_winner_cumulative_time()

        if len(winner_cumulative_time) != len(race_history_df):
            raise ValueError(
                f"Data length mismatch: DataFrame expected to be length {len(winner_cumulative_time)}  "
                f"but got {len(race_history_df)}.\nIf you are using live state updates, check you are simulating "
                f"from the 'current' lap from your current race state."
                "\nExample: If the current race is on lap 9 in the method put live_state_updates(..., current_lap=9) ")

        return race_history_df, winner_cumulative_time

    def get_predicted_finishing_positions(self, format_json: bool = True, json_indent: Optional[int | str] = None) \
            -> str | dict[str, int]:
        """Retrieves the predicted finishing positions for all drivers from the simulation.

        This method extracts the final positions of each driver from the simulation results.
        It can return the data either as a Python dictionary or as a JSON-formatted string.

        Args:
            format_json (bool): If True, returns a JSON string. If False, returns a Python dictionary.
            json_indent (int, str, or None): The indentation level for the JSON output.
                                            Use None for compact JSON, an integer for space indentation,
                                            or a string (like "\t") for custom indentation.

        Returns:
            str or dict: A JSON string or a dictionary where keys are driver names and
            values are their predicted finishing positions.

        Raises:
            AttributeError: If the simulation hasn't been run.
            ValueError: If the results are invalid

        """

        self._check_simulation_run()
        self._check_validity_simulation_result()

        result = {driver: data['position'] for driver, data in self._sim_race_state.items()}

        return json.dumps(result, indent=json_indent) if format_json else result

    def get_result_for_race_trace_plotting(self, format_json: bool = True, json_indent: Optional[int | str] = None) \
            -> str | dict[str, list[float]]:
        """
        This method returns the difference to the predicted winner for each driver.
        It can return the result either as a JSON string or as a Python dictionary.

        Args:
            format_json (bool): If True, returns a JSON string. If False, returns a Python dictionary.
            json_indent (int, str, or None): The indentation level for the JSON output.
                                            Use None for compact JSON, an integer for space indentation,
                                            or a string (like "\t") for custom indentation.

        Returns:
            str or dict: A JSON string if format_json is True, otherwise a Python dictionary.
        """

        # Exclude the winner column from subtraction
        race_history_df = self._get_race_result_df()
        # print(race_history_df)
        columns_to_subtract = race_history_df.drop('WinnerCumulativeTime', axis=1)

        time_deltas_to_winner = columns_to_subtract.sub(race_history_df['WinnerCumulativeTime'], axis=0)

        # Converting to dictionary then to JSON in the following lines, for easier data manipulation in web
        dict_format = time_deltas_to_winner.to_dict('list')

        return json.dumps(dict_format, indent=json_indent) if format_json else dict_format

    def plot_race_trace(self):
        """Generate and display an interactive race trace plot.

        This method creates a visual representation of the race progress, showing
        the time difference between each driver and the race leader over the course
        of the race.

        The plot includes:
            - Individual lines for each driver, color-coded and styled based on team.
            - Markers indicating each lap.
            - A vertical line indicating the current lap if the simulation is based on live data.

        Features:
            - Custom colors and styles for each driver/team.
            - Inverted y-axis to show leaders at the top.
            - Interactive Plotly graph for zooming and hovering over data points.
            - Dark theme for better visibility.

        Notes:
            - The plot uses the 'WinnerCumulativeTime' as a baseline for comparison.
            - If a '_virtual_line' attribute exists, it adds a vertical line to indicate
            the transition point between historical and simulated data.

        Returns:
            None. Displays the plot using Plotly's `show` method.
        """

        df = self._get_race_result_df()

        figure = go.Figure()
        laps = np.arange(1, self.race_config.num_laps + 1)

        # driver colours and styles
        driver_styles = driver_styles_for_plotting

        # Plot each driver's time difference with the winner's cumulative time
        for column in df.columns:
            if column != 'WinnerCumulativeTime':  # I'm not plotting the winner column against itself
                # Calculate the difference between the driver's times and the winner's times
                time_difference = df[column] - df['WinnerCumulativeTime']
                if column in driver_styles:
                    figure.add_trace(go.Scatter(
                        x=laps,
                        y=time_difference,
                        mode='lines+markers',
                        name=column,
                        line=dict(color=driver_styles[column]['color'], dash=driver_styles[column]['line']['dash']),
                        marker=dict(symbol=driver_styles[column]['marker'], size=8)
                    ))
                else:
                    figure.add_trace(go.Scatter(
                        x=laps,
                        y=time_difference,
                        mode='lines+markers',
                        name=column
                    ))

        if self._virtual_line is not None:
            figure.add_vline(x=self._virtual_line - 1, line_dash="dash", line_color="#FFD700", line_width=4, opacity=1)

        figure.update_layout(
            title='Race Trace Prediction',
            xaxis_title='Lap number',
            yaxis_title='Time Difference (s)',
            legend_title='Drivers',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis=dict(gridcolor='white', zerolinecolor='white'),
            yaxis=dict(gridcolor='white', zerolinecolor='white', autorange='reversed'),
            legend=dict(font=dict(size=22))
        )

        figure.show()

    def plot_tyre_usage(self):
        """Generates and displays a stacked bar chart of tyre usage for each driver in a race simulation.

        This method creates a horizontal stacked bar chart showing the tyre compounds used by each driver
        and the number of laps run on each compound. Drivers are sorted by their finishing position, with
        the winner at the top. The chart uses color coding to distinguish between different tyre compounds.

        Returns:
            None. The function displays the plot using Plotly's `show` method.

        Note:
            - If unknown compounds are encountered, colors are assigned from a predefined Plotly color scale.
        """

        race_data = self.get_result_from_simulation_run(format_json=False)

        colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'white'}

        old_colors = {'hyper_soft': 'pink',
                      'ultra_soft': 'purple',
                      'super_soft': 'red',
                      'soft': 'yellow',
                      'medium': 'white',
                      'hard': 'blue',
                      'super_hard': 'orange'}

        # Check if any driver uses a compound not in the modern colors
        use_old_colors = any(
            compound not in colors
            for driver_data in race_data.values()
            for compound in driver_data['strategy']
        )

        if use_old_colors:
            colors = old_colors

        # If there are compounds not in either color set, assign random colors
        all_compounds = set(
            compound
            for driver_data in race_data.values()
            for compound in driver_data['strategy']
        )

        color_scale = px.colors.qualitative.Plotly
        # Assign colors to compounds not in the predefined sets
        color_index = 0
        for compound in all_compounds:
            if compound not in colors:
                colors[compound] = color_scale[color_index % len(color_scale)]
                color_index += 1

        fig = go.Figure()

        sorted_drivers = sorted(race_data.keys(), key=lambda x: race_data[x]['end_position'])

        for driver in sorted_drivers:
            tyre_usage = race_data[driver]['tyre_usage']
            for i, (compound, laps) in enumerate(tyre_usage, start=1):
                fig.add_trace(go.Bar(
                    y=[driver],
                    x=[laps],
                    orientation='h',
                    marker_color=colors[compound],
                    marker=dict(
                        line=dict(width=0.5, color='black')
                    ),
                    text=f"{compound.upper()} - {laps} laps (Stint {i}) ",
                    textfont=dict(size=22),
                    textposition='inside',
                    name=f"{driver} Stint {i}",
                ))

        # Update layout
        # Update layout
        fig.update_layout(
            barmode='stack',
            title=dict(
                text='Tyre Usage by Driver',
                font=dict(size=24)  # Main title font size
            ),
            xaxis=dict(
                title='Number of Laps',
                titlefont=dict(size=22),  # X-axis title font size
                tickfont=dict(size=20)    # X-axis tick labels font size
            ),
            yaxis=dict(
                title='Driver',
                titlefont=dict(size=22),  # Y-axis title font size
                tickfont=dict(size=20),   # Y-axis tick labels font size
                categoryorder='array', 
                categoryarray=sorted_drivers[::-1]
            ),
            showlegend=False,
            legend=dict(font=dict(size=22))
        )

        # Show the figure
        fig.show()

    def __gather_results_from_custom_sim_logic(self):
        """
        This method explains how to obtain simulation results from either a 'higher fidelity'
        simulation or a custom simulation method that may offer more advanced features than
        the current implementation.

        # Note: All drivers' current strategies and tyre models are provided by self.driver_strategies and
        self.drivers_tyre_models. # You may use this to formulate each driver's performance and strategies,
        such as when they will pit, how many stops they will make, # what tyres they will use,
        etc. self.race_configuration also has information about the racetrack characteristics, # from basic details
        like the lap number to extra features that are fully customizable. # Another thing to ensure is that your
        simulation is based FULLY on the self.driver_strategies, self.drivers_tyre_models # and at a MINIMUM the
        self.race_configuration.num_laps and self.race_configuration.pit_lane_time_loss in self.race_configuration,
        this way you wouldn't have to change anything to use the MonteCarlo implementation

        # Effectively you will be creating an 'interaction' method everything else will be taken care of for you !!!
        The plotting of what occurred is # handled by other class methods, the calculation of starting strategies is
        done for you by leveraging Optimization and Combinatorics classes, # the child class
        MonteCarloRaceSimulations takes care of testing other 'alternative strategies', as it will leverage the
        Driver's class # methodology to calculate and store different strategies.

        # When complete, you may name your simulation as _simulation() to ensure it integrates seamlessly with the
        class, and delete the previous one # provided it adheres to the format of results you output when complete.
        """

        # It's crucial to transform the results from the custom simulation into a format that
        # both the parent class and its subclasses can utilize.
        # A custom simulation might segment the race into smaller sections, such as 3 or 25 mini sectors,
        # or employ time-based or distance-based discretization.
        # Regardless of the chosen method, it's essential to store the 'end of lap' times for each driver,
        # as this data is necessary for generating plots and other results.

        # Key variables for integrating custom simulation results:
        # self._sim_race_state
        # self._sim_accumulated_race_time

        # Transform your simulation's results, so they can be used by other classes:
        # _sim_race_state should include driver names as keys with their end positions and total times,
        # following the structure used in the _race_state configuration. Lap times can be omitted if not used.
        your_valid_transformed_result_dictionary = {
            'Leclerc': {'position': 2, 'total_time': 5118.3},
            'Hamilton': {'position': 4, 'total_time': 5111.94},
            'Verstappen': {'position': 1, 'total_time': 5099.5},
            'Piastri': {'position': 5, 'total_time': 5178.0},
            'Norris': {'position': 3, 'total_time': 5166.97},
        }

        # Example accumulated race time dictionary, potentially calculated using numpy's np.cumsum:
        your_valid_transformed_accumulated_race_time_dictionary = {
            "Leclerc": [96.75093000000001, 189.75279, 281.75558, 374.7593, 467.76395, 560.76953, 653.7760400000001,
                        746.78348, 839.7918500000001, ...],
            "Hamilton": [97.50093000000001, 190.50279, 280.50558, 373.5093, 465.51395, 557.51953, 650.5260400000001,
                         741.53348, 834.5418500000001, ...],
            "Verstappen": [96.00593, 189.01779, 281.03558, 373.0593, 466.08895, 558.12453, 650.1660400000001,
                           742.2134800000001, 834.2668500000001, ...],
            "Piastri": [96.25593, 189.26779, 281.28558, 373.3093, 465.33895, 558.37453, 650.4160400000001,
                        741.5634800000001, 833.6168500000001, ...],
            "Norris": [95.81093, 188.88279, 281.96558, 373.0593, 466.16395, 558.27953, 650.4060400000001, 742.54348,
                       835.69185, ...],
        }
        # TIP: In your initial simulation, you don't need to worry about tracking the accumulated lap times of each
        # driver during your 'custom' simulation. Just track their individual lap times. Numpy has a method,
        # np.cumsum, that can perform this transformation for you!

        # Note: If you intend your simulation to be "live state update" capable the simulation needs to be able to
        # take some state externally, it should be able to accept external inputs such as current positions,
        # current compounds, current lap etc. and start its simulation from that point. Your _sim_accumulated_race_time
        # values for each driver will thus be smaller than total laps in the race but don't worry the external race
        # history is ingested into the instance and is smartly used to make up what happened preceding to your
        # simulation. See the live_state_updates method for how it takes external data and includes into the instance
        # After completing the transformation of results, update the instance variables:
        self._sim_race_state = your_valid_transformed_result_dictionary
        self._sim_accumulated_race_time = your_valid_transformed_accumulated_race_time_dictionary

        """
            # Pseudocode for Running a Simulation and Processing Results

            # 1. Run your custom simulation
            # your_simulation()

            # 2. Retrieve the results from the simulation
            # results = your_simulation()

            # 3. Transform the results from the simulation
            # transformed_results = transform(results)

            # 4. Update instance variables with the transformed results
            # self._sim_race_state = your_valid_transformed_result_dictionary
            # self._sim_accumulated_race_time = your_valid_transformed_accumulated_race_time_dictionary
        """

    def __example_custom_simulation_integration(self):
        # Pseudocode for Running a Simulation and Processing Results

        # 1. Run your custom simulation
        # your_simulation()

        # 2. Retrieve the results from the simulation
        # results = your_simulation()

        # 3. Transform the results from the simulation
        # transformed_results = transform(results)

        # 4. Update instance variables with the transformed results
        # self._sim_race_state = your_valid_transformed_result_dictionary
        # self._sim_accumulated_race_time = your_valid_transformed_accumulated_race_time_dictionary

        # Example below is exactly how you would bring your results into the instance, when it is done being transformed
        # at the end of your simulation run You may comment out the _simulation() method in run_simulation() and use
        # this one, and you can see how you still have the functionality of plotting the results and view what took
        # place provided your followed __gather_results_from_custom_sim_logic
        self._sim_race_state = end_race_result_test
        self._sim_accumulated_race_time = end_race_laptimes_test


if __name__ == '__main__':
    from simulation_parameters_example import drivers, race_state, race_config
    import time

    test_race = RaceSimulation(drivers, race_state, race_config)

    # Start the timer
    start_time = time.perf_counter()
    # print(test_race.drivers_consistency)
    # print('n',test_race._get_drivers_pit_and_compound_info())

    test_race.run_simulation()
    test_race.display()

    print(test_race.get_result_from_simulation_run(format_json=False))
    # test_race.get_result_from_race_sim_run(format_json=True)
    # for drv, results in test_race.get_result_from_race_sim_run(format_json=False).items():
    #     print(drv, results)
    # test_race.run_simulation()
    # test_race.display()
    # print(test_race._driver_laps_behind_traffic)
    # test_race.update_drivers_tyre_usage()
    # test_race.plot_race_trace()

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Execution time: {duration} seconds.")

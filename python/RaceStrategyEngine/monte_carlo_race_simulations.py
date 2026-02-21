"""This module implements the MonteCarloRaceSimulations class, which extends the RaceStrategyEngine class
to perform Monte Carlo simulations for exploring various race strategies and outcomes in Formula 1.

The MonteCarloRaceSimulations class provides functionality for:
- Running multiple race simulations with varying strategies and pit stop timings
- Analyzing probabilistic race results based on different strategic variations
- Visualizing simulation results through various plots and charts
This class is particularly useful for assessing the effectiveness of different race strategies
under various conditions and for understanding the range of possible outcomes in a race."""

import random
import time
from typing import Literal
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

from .driver import Driver
from .monte_carlo_analysis import (
    plot_drivers_finishing_position_frequency,
    plot_traffic_heatmap,
    plot_traffic_boxplot,
    plot_race_strategy_position_distribution,
    plot_drivers_position_dominance,
    plot_race_strategy_ranking,
)
from .race_configuration import RaceConfiguration
from .combinatorics import StrategyCombinatorics
from .race_simulation import RaceSimulation
from .utility import count_driver_simulations, f1_radio


class MonteCarloRaceSimulations(RaceSimulation):
    """
    Extends RaceStrategyEngine to implement Monte Carlo methods for exploring various race strategies
    and outcomes. This class facilitates the running of multiple simulations to analyze probabilistic
    race results based on different strategic variations and pit stop timings.
    """

    def __init__(
        self,
        drivers: dict[str, Driver],
        starting_race_grid: dict,
        race_configuration: RaceConfiguration,
        lap_variation_range: tuple[int, int] = (-10, 10),
        simulation_type: Literal["time_discrete", "lap_discrete"] = "lap_discrete",
        track_name = 'abu_dhabi',
        time_step: float = 0.1
    ):
        super().__init__(drivers, starting_race_grid, race_configuration)

        if not (
            isinstance(lap_variation_range, tuple)
            and len(lap_variation_range) == 2
            and all(isinstance(x, int) for x in lap_variation_range)
        ):
            raise ValueError("lap_variation_range must be a tuple of two integers")

        self._lap_variation_range = lap_variation_range
        # Alternatives strategy choices are pre-computed within the Driver object with the aid of the
        # Combinatorics class to save CPU from calculating this every time.
        self.drivers_alternate_strategies = self._get_strategy_alternatives()
        self.drivers_max_stops_alternate_strategies = (
            self._get_drivers_max_stops_in_alternative_strategies()
        )
        self._max_stops_among_all_the_drivers = self._get_max_stop_among_all_drivers()

        # Precompute strategies by stops for each driver
        self.drivers_alternate_strategies_by_stops = {
            driver: self.drivers[driver].get_alternate_strategies(dict_mode=True)
            for driver in self.drivers
        }
        # For schema purposes, how the dataframe's columns will be ordered
        self._schema_order = [
            ("driver", str),
            ("compounds_used", list),
            ("pit_stop_laps", int),
            ("points", int),
            ("position", int),
            ("start_position", int),
            ("stops", int),
            ("tyre_usage", list),
            ("race_time", float),
            ("laps_behind_traffic", int),
        ]

        # Update your index lookup (remove the parenthesis instantiation)
        self._pit_stops_start_idx = self._schema_order.index(("pit_stop_laps", int))

        # Find the index where pit stop fields start and end, this for future proofing if I decide to change the
        # order of fields so the rest of my code doesn't break
        self._pit_stop_end_idx = (
            self._pit_stops_start_idx + self._max_stops_among_all_the_drivers
        )

        self._drivers_pit_stop_indices = self._get_drivers_pit_stop_index()
        self._enable_exp_rust_backend(simulation_type, track_name, time_step)

    

    def _get_drivers_max_stops_in_alternative_strategies(self) -> dict:
        """
        Get the longest strategy each driver has in their alternative strategies list.

        Returns:
            (dict): A dictionary mapping driver names to their maximum number of stops in alternative strategies.
        """

        return {
            name: driver_object.highest_amount_of_stops_in_alternate_strategies
            for name, driver_object in self.drivers.items()
        }

    def _get_max_stop_among_all_drivers(self) -> int:
        """Get the maximum stop amongst all drivers

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
        self.drivers_max_stops_alternate_strategies = (
            self._get_drivers_max_stops_in_alternative_strategies()
        )
        self._max_stops_among_all_the_drivers = self._get_max_stop_among_all_drivers()

        # Heavy
        self.drivers_alternate_strategies_by_stops = {
            driver: self.drivers[driver].get_alternate_strategies(dict_mode=True)
            for driver in self.drivers
        }

    def _get_strategy_alternatives(self) -> dict:
        """
        Retrieve alternate strategy choices for all drivers and organize them into a dictionary.

        Returns:
            (dict): A dictionary mapping driver names to their list of alternative strategies.
        """

        return {
            name: driver_object.get_alternate_strategies()
            for name, driver_object in self.drivers.items()
        }

    def _select_random_strategy(self, driver):
        """Randomly select and modify a strategy for a driver.

        This method selects an alternate strategy for the given driver, shuffles the order
        of the compounds, and modifies the number of laps for each stint within a predefined
        range. The resulting modified strategy is then assigned to the driver.

        Args:
            driver (str): The name of the driver for whom to select a random strategy.
        """

        #  TODO be able to change how strategies are sampled

        alternate_strategy = self._select_alternate_strategy(driver)

        shuffled_strategy = self._shuffle_strategy(alternate_strategy)

        modified_strategy = self._modify_strategy(
            driver=driver,
            strategy=shuffled_strategy,
            lap_variation_range=self._lap_variation_range,
        )

        self.driver_strategies[driver] = modified_strategy

    def _get_alternate_strategies_rust(self) -> tuple[list, dict[str, list]]:
        sim_data = self._get_rust_simulation_data()

        drivers_alternate_strategies_per_stop_rust = {
            driver: {} for driver in self._race_state
        }

        for (
            driver,
            strategies_by_stops,
        ) in self.drivers_alternate_strategies_by_stops.items():
            for stop, strategies in strategies_by_stops.items():
                drivers_alternate_strategies_per_stop_rust[driver][stop] = []
                for strategy in strategies:
                    compounds, laps_on_compounds = (
                        self._extract_strategy_compounds_and_laps(
                            optional_strategy=strategy, included_race_time=True
                        )
                    )

                    rounded_laps_on_compounds = self._rusty_smart_round(
                        laps_on_compounds, self.race_config.num_laps, False
                    )

                    strategy = list(zip(compounds, rounded_laps_on_compounds))

                    drivers_alternate_strategies_per_stop_rust[driver][stop].append(
                        strategy
                    )

        return (sim_data, drivers_alternate_strategies_per_stop_rust)

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

        strategies = self.drivers_alternate_strategies_by_stops[driver]

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

        _, *compound_strategy = strategy
        compounds, laps_on_compounds = zip(*compound_strategy)

        # Randomly adjust the number of laps for each compound
        modified_laps = self._adjust_laps_randomly(
            list(laps_on_compounds), lap_variation_range
        )

        # Construct the new modified strategy
        modified_lap_dist = tuple(zip(compounds, modified_laps))
        driver_object = self.drivers[driver]
        new_strategy = driver_object.strategy_creator(
            new_lap_distributions=modified_lap_dist
        )

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
        modified_laps = [
            max(1, laps + random.randint(*lap_variation_range))
            for laps in laps_on_compounds
        ]

        # Check if any compound is assigned less than 1 lap or total laps doesn't equal number of race laps
        if (
            any(lap < 1 for lap in modified_laps)
            or sum(modified_laps) != self.race_config.num_laps
        ):
            # applying smart round to fix the error
            modified_laps = StrategyCombinatorics.smart_round(
                modified_laps, self.race_config.num_laps, scale_by_contribution=True
            )

        return modified_laps  # Return the modified laps when valid

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
            driver_max_amount_of_stops = self.drivers_max_stops_alternate_strategies[
                driver
            ]

            driver_pit_stops_start_idx = self._schema_order.index(
                ("pit_stop_laps", int)
            )
            driver_pit_stop_end_idx = (
                driver_pit_stops_start_idx + driver_max_amount_of_stops
            )

            driver_pit_index[driver] = (
                driver_pit_stops_start_idx,
                driver_pit_stop_end_idx,
            )

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
            if field == "pit_stop_laps":
                for i in range(1, max_amount_of_stops + 1):
                    pit_lap_keys = f"pit_stop_{i}_lap"
                    fields.append(pit_lap_keys)
                    schema_fields.append((pit_lap_keys, field_type))
            else:
                fields.append(field)
                schema_fields.append((field, field_type))

        return fields, tuple(schema_fields)

    def run_monte_carlo_simulation(self):
        """Execute a Monte Carlo simulation of a race.

        This method orchestrates a Monte Carlo simulation of a race by iteratively modifying
        each driver's strategy, shuffling it, and adjusting it based on a predefined range
        of lap variations. After all modifications, it triggers the actual race simulation.

        Args:
            adaptive (bool, optional): If True, the simulation uses adaptive constraints based
                on the current race state. Defaults to False.
        """

        for driver in self._race_state:
            self._select_random_strategy(driver)

        self.run_simulation()

    def _check_simulation_run(self, custom_message=None):
        message = (
            "Simulation state is not initialized. Please run the simulation first by utilizing the"
            " 'run_monte_carlo_simulation()' method\n. If you would like to avoid the stochastic effects"
            " of the monte carlo you may use 'run_simulation()' instead."
        )
        super()._check_simulation_run(custom_message=message)

    def run_and_analyze_simulations_from_monte_carlo_runs(
        self,
        num_simulations: int,
        method: Literal[
            "single_core",
            "multi_core",
        ] = "multi_core",
    ):
        self._check_exp_backend_on()

        simulation_data, alternate_strategies = self._get_alternate_strategies_rust()

        # print("...", end='\n\n')

        message = "\nI need to wait until all simulations are done. I need to wait.\nBut it's looking good!"
        message += "\nYou just wait sunshine, you just wait..."
        f1_radio("Race Engineer", message)

        start = time.perf_counter()
        max_stops = self._max_stops_among_all_the_drivers
        result = self._simulation_engine.run_monte_carlo_simulations(
            simulation_data, alternate_strategies, max_stops, num_simulations, method
        )
        end = time.perf_counter()
        print(
            f"\nRust backend for {num_simulations:_} simulations Execution time: {end - start} "
            + "(NOTE: Time includes time to bring results over to python.)\n"
        )

        message = "\nGet in there! You are the man!\n\nYou.\n\nAre.\n\nThe.\n\nMan.\n"
        message += "\nYou knocked that out the park today mate. Out the park.\n"

        message = "Du bist WELTMEISTER!"

        f1_radio("Race Engineer", message)

        result = self._create_df_from_monte_carlo_results(results=result)

        return result

    def _create_df_from_monte_carlo_results(self, results) -> pd.DataFrame:
        column_order = [
            "driver",
            "compounds_used",
            # pit stops go here dynamically
            "points",
            "position",
            "start_positions",
            "stops",
            "tyre_usage",
            "race_time",
            "laps_behind_traffic",
        ]

        data = {
            "driver": results.names,
            "compounds_used": results.compounds_used,
            "points": list(results.points),
            "position": list(results.positions),
            "start_positions": list(results.start_positions),
            "stops": list(results.amount_of_stops),
            "tyre_usage": results.strategies,
            "race_time": results.race_times,
            "laps_behind_traffic": list(results.laps_behind_traffic),
        }

        for i, lap_data in enumerate(results.laps_pitted_on):
            data[f"pit_stop_{i + 1}_lap"] = list(lap_data)

        # TODO fix codebase so i dont have ot order coloumns like this
        # Build final column order by inserting pit stops after 'compounds_used' for ordering
        final_columns = column_order[:2]  # up to 'compounds_used'
        final_columns.extend(
            [f"pit_stop_{i + 1}_lap" for i in range(len(results.laps_pitted_on))]
        )
        final_columns.extend(column_order[2:])  # rest of the columns

        df = pd.DataFrame(data)[final_columns]

        # Making compounds used a string so its hashable and uniqueness can be indentified
        df["compounds_used_list"] = df["compounds_used"]
        df["compounds_used"] = df["compounds_used"].astype(str)

        return df

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

        drivers = pandas_df["driver"].unique()
        fig = go.Figure()
        colors = plotly.colors.qualitative.Plotly

        for driver in drivers:
            driver_df = pandas_df[pandas_df["driver"] == driver]
            fields = self._generate_field_names()

            # All drivers have the same index for when their pit stops start
            first_pit_lap = fields[self._pit_stops_start_idx]

            for i, strategy in enumerate(driver_df["compounds_used"].unique()):
                subset = driver_df[driver_df["compounds_used"] == strategy]
                x = subset[first_pit_lap].values
                y = subset["position"].values  # Changed from 'points' to 'position'

                coefficients = np.polyfit(x, y, 2)
                polynomial = np.poly1d(coefficients)

                x_fit = np.linspace(x.min(), x.max(), 1000)
                y_fit = polynomial(x_fit)

                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name=f"{strategy} ({driver})",
                        line=dict(color=colors[i % len(colors)]),
                        visible=(driver == drivers[0]),
                    )
                )

        dropdown_buttons = [
            dict(
                method="update",
                label=driver,
                args=[
                    {"visible": [driver in trace.name for trace in fig.data]},
                    {
                        "title": f"Pre-Event Race Strategy Evaluations for {driver} (Quadratic Fit)"
                        f" ({count_driver_simulations(pandas_df, driver):,} Simulations)"
                    },
                ],
            )
            for driver in drivers
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    x=1.0,
                    y=1.1,
                    xanchor="right",
                    yanchor="top",
                )
            ],
            title_text=f"Pre-Event Race Strategy Evaluations for {drivers[0]} (Quadratic Fit)"
            f" ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
            xaxis_title="Pit Lap (first stop)",
            yaxis_title="Mean Position",  # Changed from 'Mean Points' to 'Mean Position'
            yaxis=dict(
                autorange="reversed"
            ),  # Reverse y-axis so lower positions (better) are at the top
            legend_title="Race Strategy",
            legend=dict(
                x=1.05,
                y=1,
                orientation="v",
            ),
            margin=dict(l=40, r=40, t=40, b=40),
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

        drivers = pandas_df["driver"].unique()
        fig = go.Figure()

        position_min = pandas_df["position"].min()
        position_max = pandas_df["position"].max()

        custom_colors = ["purple", "pink", "#32CD32", "#FFFF00", "#DC0000"]

        for driver in drivers:
            driver_df = pandas_df[pandas_df["driver"] == driver]
            fields = self._generate_field_names()
            driver_pit_stops_start_idx, driver_pit_stop_end_idx = (
                self._drivers_pit_stop_indices[driver]
            )
            driver_pit_lap_fields = fields[
                driver_pit_stops_start_idx:driver_pit_stop_end_idx
            ]

            dimensions = [
                *[
                    dict(label=field.replace("_", " ").title(), values=driver_df[field])
                    for field in driver_pit_lap_fields
                ],
                dict(
                    label="Position",
                    values=driver_df["position"],
                    range=[driver_df["position"].min(), driver_df["position"].max()],
                ),
            ]

            fig.add_trace(
                go.Parcoords(
                    line=dict(
                        color=driver_df["position"],
                        colorscale=custom_colors,
                        showscale=True,
                        colorbar=dict(title="Position", x=1.05, y=0.5),
                        cmin=position_min,
                        cmax=position_max,
                    ),
                    dimensions=dimensions,
                    name=driver,
                    visible=(driver == drivers[0]),
                )
            )

        dropdown_buttons = [
            dict(
                method="update",
                label=driver,
                args=[
                    {"visible": [driver == trace.name for trace in fig.data]},
                    {
                        "title": f"Parallel Coordinate Plot of Race Strategy Simulation Results for {driver} "
                        f"({count_driver_simulations(pandas_df, driver):,} Simulations)"
                    },
                ],
            )
            for driver in drivers
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    x=1.0,
                    y=1.1,
                    xanchor="right",
                    yanchor="top",
                )
            ],
            title_text=f"Parallel Coordinate Plot of Race Strategy Simulation Results for {drivers[0]}"
            f" ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
            showlegend=False,
            margin=dict(r=150),
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
        drivers = pandas_df["driver"].unique()
        fig = go.Figure()

        position_min = pandas_df["position"].min()
        position_max = pandas_df["position"].max()

        custom_colors = ["purple", "pink", "#32CD32", "#FFFF00", "#DC0000"]

        jitter = (0.1, 0.3)
        lower_bound, upper_bound = jitter

        for driver in drivers:
            driver_df = pandas_df[pandas_df["driver"] == driver].copy()

            fields = self._generate_field_names()
            driver_pit_stops_start_idx, driver_pit_stop_end_idx = (
                self._drivers_pit_stop_indices[driver]
            )
            driver_pit_lap_fields = fields[
                driver_pit_stops_start_idx:driver_pit_stop_end_idx
            ]

            jittered_fields = [f"{field}_jittered" for field in driver_pit_lap_fields]
            driver_df[jittered_fields] = driver_df[driver_pit_lap_fields].apply(
                lambda x: x + np.random.uniform(lower_bound, upper_bound, x.shape)
            )

            hover_text = [
                f"Driver: {driver}<br>"
                + f"Strategy: {row['compounds_used']}<br>"
                + f"Points: {row['points']}<br>"
                + f"Position: {row['position']}<br>"
                + f"Laps Behind Traffic: {row['laps_behind_traffic']}<br>"
                + f"First Pit Stop: {row[jittered_fields[0]]:.2f}<br>"
                + (
                    f"Second Pit Stop: {row[jittered_fields[1]]:.2f}<br>"
                    if len(jittered_fields) > 1
                    else ""
                )
                + (
                    f"Third Pit Stop: {row[jittered_fields[2]]:.2f}<br>"
                    if len(jittered_fields) > 2
                    else ""
                )
                for _, row in driver_df.iterrows()
            ]

            x = driver_df[jittered_fields[0]]
            y = (
                driver_df[jittered_fields[1]]
                if len(jittered_fields) > 1
                else [0] * len(x)
            )
            z = (
                driver_df[jittered_fields[2]]
                if len(jittered_fields) > 2
                else [0] * len(x)
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=driver_df["position"],
                        colorscale=custom_colors,
                        opacity=0.8,
                        colorbar=dict(title="Positions Achieved"),
                        cmin=position_min,
                        cmax=position_max,
                        showscale=True,
                    ),
                    text=hover_text,
                    hoverinfo="text",
                    name=driver,
                    visible=(driver == drivers[0]),
                )
            )

        dropdown_buttons = [
            dict(
                method="update",
                label=driver,
                args=[
                    {"visible": [driver == trace.name for trace in fig.data]},
                    {
                        "title": f"3D Scatter Plot of Race Strategy Simulation Results (Jitter range of {jitter})"
                        f" for {driver} ({count_driver_simulations(pandas_df, driver):,} Simulations)"
                    },
                ],
            )
            for driver in drivers
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    x=1.0,
                    y=1.1,
                    xanchor="right",
                    yanchor="top",
                )
            ],
            title_text=f"3D Scatter Plot of Race Strategy Simulation Results (Jitter range of {jitter})"
            f" for {drivers[0]} ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
            scene=dict(
                xaxis_title="First Pit Stop (jittered)",
                yaxis_title="Second Pit Stop (jittered)",
                zaxis_title="Third Pit Stop (jittered)",
            ),
            showlegend=False,
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
        drivers = pandas_df["driver"].unique()
        fig = go.Figure()
        position_min = pandas_df["position"].min()
        position_max = pandas_df["position"].max()

        custom_colors = ["purple", "pink", "#32CD32", "#FFFF00", "#DC0000"]

        for driver in drivers:
            driver_df = pandas_df[pandas_df["driver"] == driver]
            fields = self._generate_field_names()
            driver_pit_stops_start_idx, driver_pit_stop_end_idx = (
                self._drivers_pit_stop_indices[driver]
            )
            driver_pit_lap_fields = fields[
                driver_pit_stops_start_idx:driver_pit_stop_end_idx
            ]

            hover_text = [
                f"Driver: {driver}<br>"
                + f"Strategy: {row['compounds_used']}<br>"
                + f"Points: {row['points']}<br>"
                + f"Position: {row['position']}<br>"
                + f"Laps Behind Traffic: {row['laps_behind_traffic']}<br>"
                + f"First Pit Stop: {row[driver_pit_lap_fields[0]]}<br>"
                + (
                    f"Second Pit Stop: {row[driver_pit_lap_fields[1]]}<br>"
                    if len(driver_pit_lap_fields) > 1
                    else ""
                )
                + (
                    f"Third Pit Stop: {row[driver_pit_lap_fields[2]]}<br>"
                    if len(driver_pit_lap_fields) > 2
                    else ""
                )
                for _, row in driver_df.iterrows()
            ]
            x = driver_df[driver_pit_lap_fields[0]]
            y = (
                driver_df[driver_pit_lap_fields[1]]
                if len(driver_pit_lap_fields) > 1
                else [0] * len(x)
            )
            z = (
                driver_df[driver_pit_lap_fields[2]]
                if len(driver_pit_lap_fields) > 2
                else [0] * len(x)
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=driver_df["position"],
                        colorscale=custom_colors,
                        opacity=0.8,
                        colorbar=dict(
                            title="Positions Achieved",  # Format tic
                        ),
                        cmin=position_min,
                        cmax=position_max,
                        showscale=True,
                        # reversescale=True
                    ),
                    text=hover_text,
                    hoverinfo="text",
                    name=driver,
                    visible=(driver == drivers[0]),
                )
            )

        dropdown_buttons = [
            dict(
                method="update",
                label=driver,
                args=[
                    {"visible": [driver == trace.name for trace in fig.data]},
                    {
                        "title": f"3D Scatter Plot of Race Strategy Simulation Results (Basic) for {driver}"
                        f" ({count_driver_simulations(pandas_df, driver):,} Simulations)"
                    },
                ],
            )
            for driver in drivers
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    x=1.0,
                    y=1.1,
                    xanchor="right",
                    yanchor="top",
                )
            ],
            title_text=f"3D Scatter Plot of Race Strategy Simulation Results (Basic) for {drivers[0]}"
            f" ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
            scene=dict(
                xaxis_title="First Pit Stop",
                yaxis_title="Second Pit Stop",
                zaxis_title="Third Pit Stop",
            ),
        )

        fig.show()


if __name__ == "__main__":
    pass
    # NOTE to self: When running this file directly as a module (python -m RaceStrategyEngine.monte_carlo_race_simulations),
    # PySpark has difficulty accessing class methods in worker processes. Best to test in another file and call it

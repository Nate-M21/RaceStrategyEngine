"""This module defines the RaceConfiguration class, which encapsulates all the configuration parameters for a race
simulation. The RaceConfiguration dataclass is a comprehensive data structure that holds various race characteristics
and rules, including track properties, pit lane time loss, overtaking thresholds or probabilities, overtaking
interactions, time losses, effect of race grid starting position, point distribution etc. 
It serves as a central configuration object for the RaceStrategyEngine and Driver, ensuring consistency across
different components of the simulation system."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sympy import Piecewise, lambdify, symbols

from .utility import format_overtake_piecewise
from .optimisation import StrategyOptimization

# I am using the data class in this way to combine the simplicity and flexibility of Python with the rationale behind
# C structs

# Testing using weights:
# Weighting factors to reduce the impact of sampling bias and outliers
# Without weighting, strategies tested fewer times can appear falsely optimal due to "lucky" results
# Weighting multiple factors (position, race time, stops) provides more robust strategy evaluation
# while requiring fewer total simulations too to get good strategy representation
@dataclass(frozen=True, kw_only=True)
class RaceStrategyWeighting:
    position_weighting: float
    race_time_weighting: float
    amount_of_stops_weighting: float
    # (assuming a typical F1 race time of around 5400 seconds ie 90 minutes)
    estimated_total_race_time: float = 5400

@dataclass(frozen=True, kw_only=True)
class RaceConfiguration:
    """
    Represents the configuration for a race, including all relevant parameters and characteristics.

    Attributes:
        points_distribution (Dict[int, int]): Mapping of finishing positions to points awarded.
        pit_lane_time_loss (float): Fixed time lost for driving through the pit lane in seconds.
        num_laps (int): Total number of laps in the race.
        total_fuel (float): Total fuel available at the start of the race.
        fuel_effect_seconds_per_kg (float): Impact of fuel load on lap time.
        fuel_consumption_per_lap (float): Fuel consumption per lap.
        drs_boost (float): Time advantage gained from DRS activation in seconds.
        drs_activation_lap (int): Lap number from which DRS becomes available.
        time_lost_due_to_being_overtaken (float): Time lost by a driver when being overtaken.
        time_lost_performing_overtake (float): Time lost by a driver when attempting an overtake.
        min_time_lost_due_to_failed_overtake_attempt (float): Minimum time lost in a failed overtake attempt.
        max_time_lost_due_to_failed_overtake_attempt (float): Maximum time lost in a failed overtake attempt.
        overtake_chances (Callable[[float], float]): Function to calculate overtake probability based on time gap.
        race_start_stationary_time_penalty (float): Time penalty for starting from a stationary position.
        race_start_grid_position_time_penalty (float): Time penalty based on starting grid position.

    Optional Attributes: They optional because I haven't gotten around to incorporating them
        full_course_yellow_pit loss (Optional[int]): Pit loss time during a full course yellow, if applicable.
        probability_of_full_course_yellow (Optional[float]): Probability of a full course yellow occurring.
        track_evolution (Optional[Callable]): Function to model track evolution over time, if applicable. 
    """

    points_distribution: dict[int, int]
    pit_lane_time_loss: float
    num_laps: int
    drs_boost: float
    drs_activation_lap: int
    time_lost_due_to_being_overtaken: float
    time_lost_performing_overtake: float
    min_time_lost_due_to_failed_overtake_attempt: float
    max_time_lost_due_to_failed_overtake_attempt: float
    overtake_chances_piecewise: Piecewise
    # This below will be the lambdified version of overtake_chances_piecewise for fast computation in the simulation
    overtake_chances: Callable = field(init=False)
    race_start_stationary_time_penalty: float
    race_start_grid_position_time_penalty: float
    total_fuel: float
    delta_for_drs_activation: Optional[float] = 1.0
    fuel_consumption_per_lap: Optional[float] = None
    # 10 kg roughly equates to 0.3s
    fuel_effect_seconds_per_kg: float = 0.03

    # Still to implement
    probability_of_full_course_yellow: Optional[float] = None
    track_evolution: Optional[Callable] = None  # I will probably not do this

    def __post_init__(self):
        default_consumption = self.total_fuel / self.num_laps

        x = symbols('x')
        object.__setattr__(self, 'overtake_chances', lambdify(x, self.overtake_chances_piecewise,
                                                              modules=[{"Piecewise": np.piecewise}, "numpy"]))

        if self.fuel_consumption_per_lap is None:
            object.__setattr__(self, 'fuel_consumption_per_lap', default_consumption)

        # Check that all probabilities in overtake_chances_piecewise are between 0 and 1
        for (probability, _) in self.overtake_chances_piecewise.args:
            if not (0 <= float(probability) <= 1):
                raise ValueError(f"Overtake chance of {probability} is not between 0 and 1. "
                                 f"This value falls outside the valid range of 0% to 100%.")

        if self.fuel_consumption_per_lap > default_consumption:
            raise ValueError(f"Fuel consumption per lap cannot exceed average fuel available per lap.\n The maximum "
                             f"value is {default_consumption:.3f} based on the total fuel and number of laps entered."
                             f"Any higher value would lead to running out of fuel before completing the race.")

        if self.min_time_lost_due_to_failed_overtake_attempt > self.max_time_lost_due_to_failed_overtake_attempt:
            raise ValueError('The minimum time lost due to an overtake attempt that failed can not be '
                             'greater than the maximum amount')

        if self.time_lost_performing_overtake > self.time_lost_due_to_being_overtaken:
            raise ValueError("The driver performing the overtake can't lose more time than the driver being overtaken")

        if self.drs_boost > 2:
            raise ValueError('The time gained by using DRS typically does not gain you north of 1.5s, '
                             'Values above 2.0s are highly unrealistic.')

        if self.fuel_effect_seconds_per_kg > 0.05:
            raise ValueError("Fuel effect is typically around 0.03s/kg. 10kg roughly equates to 0.3s. "
                             "Values above 0.05s/kg are unrealistic.")

        StrategyOptimization.validate_pit_loss(self.pit_lane_time_loss)

    def to_dict(self) -> dict:
        """ Get a dict of all the data class attributes

        Returns:
            dict: A dictionary of the instance attributes
        """
        return {key: value for key, value in self.__dict__.items()}

    def __repr__(self):
        return (
            f"RaceConfiguration(\n"
            f"\tnum_laps: {self.num_laps}\n"
            f"\tpit_lane_time_loss: {self.pit_lane_time_loss:.2f}s\n"
            f"\tdrs_boost: {self.drs_boost:.2f}s\n"
            f"\tdrs_activation_lap: {self.drs_activation_lap}\n"
            f"\ttime_lost_due_to_being_overtaken: {self.time_lost_due_to_being_overtaken:.2f}s\n"
            f"\ttime_lost_due_to_performing_overtake: {self.time_lost_performing_overtake:.2f}s\n"
            f"\tmin_time_lost_due_to_failed_overtake: {self.min_time_lost_due_to_failed_overtake_attempt:.2f}s\n"
            f"\tmax_time_lost_due_to_failed_overtake: {self.max_time_lost_due_to_failed_overtake_attempt:.2f}s\n"
            f"\trace_start_stationary_time_penalty: {self.race_start_stationary_time_penalty:.2f}s\n"
            f"\trace_start_grid_position_time_penalty: {self.race_start_grid_position_time_penalty:.2f}s\n"
            f"\ttotal_fuel: {self.total_fuel:.2f}kg\n"
            f"\tfuel_consumption_per_lap: {self.fuel_consumption_per_lap:.2f}kg/lap\n"
            f"\tfuel_effect_seconds_per_kg: {self.fuel_effect_seconds_per_kg:.3f}s/kg\n"
            f"\tpoints_distribution: {self.points_distribution}\n"
            f"\tovertake_chances: {self.overtake_chances_piecewise}\n"
            f"\n\t{'=' * 40}\n"
            f"\tPiecewise of Overtake Chances Meaning:\n"
            f"{format_overtake_piecewise(self.overtake_chances_piecewise)}"
            f")"
        )




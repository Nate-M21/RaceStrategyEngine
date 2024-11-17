"""Configuration and data for race simulation. Effectively, the given characteristics of a race track that you can
configure and more.

This module contains the essential parameters, configurations, and initial
conditions for a race simulation. It serves as a central repository
for all the data needed to initialize and run a race simulation.

The module includes: - Race configuration (laps, DRS strength ,time lost in the pit lane, fuel consumption,
overtaking probability model, etc.) - Driver specific performance (tyre degradation model, mean deviation from ideal
lap time, mean time for tyre change pitstop, etc) - Starting grid information
    

This data is used to initialize Driver objects and the RaceStrategyEngine class,
providing a comprehensive setup for simulating a race with realistic
parameters and variability.

Typical usage example:

    from simulation_parameters import race_configuration, drivers, _race_state
    
    simulation = RaceStrategyEngine(drivers, _race_state, race_configuration)
    simulation.run_simulation()

Note:
    Modify the parameters in this file to simulate different scenarios or to
    reflect up-to-date data for specific races or seasons.
"""

from sympy import symbols, Piecewise

from RaceStrategyEngine.driver import Driver
from RaceStrategyEngine.race_configuration import RaceConfiguration, RaceStrategyWeighting
from RaceStrategyEngine.tyre_model import TyreModel

x = symbols('x')

points_mapping = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}  # this is per current regs

# Define the overtake chances function using sympy.Piecewise

overtake_chances_piecewise = Piecewise(
    (0, x < 1.1),
    (1, True)
)

# Create an instance of RaceConfig
race_config = RaceConfiguration(
    points_distribution=points_mapping,
    pit_lane_time_loss=24,
    probability_of_full_course_yellow=None,
    num_laps=55,
    total_fuel=100,
    fuel_effect_seconds_per_kg=0.033,
    fuel_consumption_per_lap=1.79,
    drs_boost=1,
    drs_activation_lap=3,
    time_lost_due_to_being_overtaken=0.6,
    time_lost_performing_overtake=0.1,
    min_time_lost_due_to_failed_overtake_attempt=0.1,
    max_time_lost_due_to_failed_overtake_attempt=0.6,
    overtake_chances_piecewise=overtake_chances_piecewise,
    race_start_stationary_time_penalty=2.5,
    race_start_grid_position_time_penalty=1.0,
)

strategy_weighting = RaceStrategyWeighting(
    position_weighting=0.7,
    amount_of_stops_weighting=0.2,
    race_time_weighting=0.1,
    estimated_total_race_time=5400
    
)

# Define the drivers performance
drivers_tyre_models = {
    "Verstappen": {"soft": TyreModel(89.44 + 0.1 * x), "medium": TyreModel(89.7 + 0.06 * x), "hard": TyreModel(90 + 0.03 * x)},
    "Perez": {"soft": TyreModel(89.44 + 0.1 * x), "medium": TyreModel(89.7 + 0.06 * x), "hard": TyreModel(90 + 0.03 * x)},
    "Hamilton": {"soft": TyreModel(89.5 + 0.11 * x), "medium": TyreModel(89.7 + 0.065 * x), "hard": TyreModel(90.2 + 0.03 * x)},
    "Russell": {"soft": TyreModel(89.5 + 0.11 * x), "medium": TyreModel(89.7 + 0.065 * x), "hard": TyreModel(90.2 + 0.03 * x)},
    "Leclerc": {"soft": TyreModel(89.3 + 0.12 * x), "medium": TyreModel(89.75 + 0.07 * x), "hard": TyreModel(90.2 + 0.04 * x)},
    "Sainz": {"soft": TyreModel(89.3 + 0.12 * x), "medium": TyreModel(89.75 + 0.07 * x), "hard": TyreModel(90.2 + 0.04 * x)},
    "Gasly": {"soft": TyreModel(89.7 + 0.13 * x), "medium": TyreModel(90.2 + 0.09 * x), "hard": TyreModel(90.7 + 0.05 * x)},
    "Ocon": {"soft": TyreModel(89.7 + 0.13 * x), "medium": TyreModel(90.2 + 0.09 * x), "hard": TyreModel(90.7 + 0.05 * x)},
    "Norris": {"soft": TyreModel(89.9 + 0.14 * x), "medium": TyreModel(90.4 + 0.1 * x), "hard": TyreModel(90.9 + 0.05 * x)},
    "Piastri": {"soft": TyreModel(89.9 + 0.14 * x), "medium": TyreModel(90.4 + 0.1 * x), "hard": TyreModel(90.9 + 0.05 * x)},
    "Alonso": {"soft": TyreModel(90.1 + 0.15 * x), "medium": TyreModel(90.6 + 0.11 * x), "hard": TyreModel(91.1 + 0.06 * x)},
    "Stroll": {"soft": TyreModel(90.1 + 0.15 * x), "medium": TyreModel(90.6 + 0.11 * x), "hard": TyreModel(91.1 + 0.06 * x)},
    "Sargeant": {"soft": TyreModel(90.3 + 0.16 * x), "medium": TyreModel(90.8 + 0.12 * x), "hard": TyreModel(91.3 + 0.06 * x)},
    "Albon": {"soft": TyreModel(90.3 + 0.16 * x), "medium": TyreModel(90.8 + 0.12 * x), "hard": TyreModel(91.3 + 0.06 * x)},
    "Zhou": {"soft": TyreModel(90.4 + 0.17 * x), "medium": TyreModel(90.9 + 0.13 * x), "hard": TyreModel(91.4 + 0.07 * x)},
    "Bottas": {"soft": TyreModel(90.4 + 0.17 * x), "medium": TyreModel(90.9 + 0.13 * x), "hard": TyreModel(91.4 + 0.07 * x)},
    "Magnussen": {"soft": TyreModel(90.6 + 0.18 * x), "medium": TyreModel(91.0 + 0.14 * x), "hard": TyreModel(91.5 + 0.07 * x)},
    "Hulkenberg": {"soft": TyreModel(90.6 + 0.18 * x), "medium": TyreModel(91.0 + 0.14 * x), "hard": TyreModel(91.5 + 0.07 * x)},
    "Ricciardo": {"soft": TyreModel(90.8 + 0.19 * x), "medium": TyreModel(91.2 + 0.15 * x), "hard": TyreModel(91.7 + 0.08 * x)},
    "Tsunoda": {"soft": TyreModel(90.8 + 0.19 * x), "medium": TyreModel(91.2 + 0.15 * x), "hard": TyreModel(91.7 + 0.08 * x)}
}

# drivers_consistency: Driver-specific parameters for lap time variation and pit stop performance
# Keys:
#   mean: Average deviation from ideal lap time (in seconds) ie their variability in lap times
#   std_dev: Standard deviation for lap time variability (in seconds)
#   min_lap_time_variation: Minimum variation applied to each lap time (in seconds)
#   mean_tyre_change_time: Average time taken to change tyres during a pit stop (in seconds)
#   std_dev_tyre_change_time: Standard deviation of tyre change time (in seconds)

drivers_consistency = {
    "Verstappen": {"mean": 0.1, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 1.9, "std_dev_tyre_change_time": 0.1},
    "Perez": {"mean": 0.15, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.1, "std_dev_tyre_change_time": 0.15},
    "Hamilton": {"mean": 0.1, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.0, "std_dev_tyre_change_time": 0.1},
    "Russell": {"mean": 0.14, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.0, "std_dev_tyre_change_time": 0.1},
    "Leclerc": {"mean": 0.13, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.1, "std_dev_tyre_change_time": 0.15},
    "Sainz": {"mean": 0.15, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.1, "std_dev_tyre_change_time": 0.15},
    "Gasly": {"mean": 0.16, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.2, "std_dev_tyre_change_time": 0.2},
    "Ocon": {"mean": 0.16, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.2, "std_dev_tyre_change_time": 0.2},
    "Norris": {"mean": 0.0, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.1, "std_dev_tyre_change_time": 0.15},
    "Piastri": {"mean": 0.14, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.2, "std_dev_tyre_change_time": 0.2},
    "Alonso": {"mean": 0.1, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.1, "std_dev_tyre_change_time": 0.15},
    "Stroll": {"mean": 0.21, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.2, "std_dev_tyre_change_time": 0.2},
    "Sargeant": {"mean": 0.21, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.3, "std_dev_tyre_change_time": 0.25},
    "Albon": {"mean": 0.16, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.2, "std_dev_tyre_change_time": 0.2},
    "Zhou": {"mean": 0.20, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.3, "std_dev_tyre_change_time": 0.25},
    "Bottas": {"mean": 0.16, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.2, "std_dev_tyre_change_time": 0.2},
    "Magnussen": {"mean": 0.18, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.3, "std_dev_tyre_change_time": 0.25},
    "Hulkenberg": {"mean": 0.17, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.3, "std_dev_tyre_change_time": 0.25},
    "Ricciardo": {"mean": 0.15, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.2, "std_dev_tyre_change_time": 0.2},
    "Tsunoda": {"mean": 0.19, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 2.3, "std_dev_tyre_change_time": 0.25},
}

# Starting grid effectively below

# Notes:
# - 'used_tyres' keeps track of all tyre stints in order, allowing for multiple stints on the same compound.
# - 'current_tyre' always reflects the compound currently in use.
# - This structure allows for detailed strategy calculations and adjustments during the race.
# - The strategy optimization can use this information to calculate the best strategy for the remaining laps.

# When updating strategies, the system can use the 'starting_compound','used_tyres' and 'current_tyre' information
# to prune impossible , focusing only on viable options for the
# remainder of the race. This is a form of optimization similar to alpha-beta pruning in concept.
# NB: At the start of the race the current tyre is the starting compound.
race_state = {
    'Leclerc': {'position': 2, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Hamilton': {'position': 4, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Verstappen': {'position': 1, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Gasly': {'position': 5, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Norris': {'position': 3, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Ocon': {'position': 6, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Piastri': {'position': 7, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Perez': {'position': 8, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    'Sainz': {'position': 9, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 53},
    'Russell': {'position': 10, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Alonso": {'position': 11, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft','current_tyre_laps_age': 0},
    "Stroll": {'position': 12, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Sargeant": {'position': 13, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Albon": {'position': 14, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Zhou": {'position': 15, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Bottas": {'position': 16, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Magnussen": {'position': 17, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Hulkenberg": {'position': 18, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Ricciardo": {'position': 19, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
    "Tsunoda": {'position': 20, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'soft', 'current_tyre_laps_age': 0},
}

# Example if say Verstappen was in 71 lap race started on on soft, used them 20 laps and then used mediums for
# 25 and is now using the hard At this point, _race_state['Verstappen'] would look like:
# {
#     'position': 1,
#     'total_time': ...,
#     'lap_time': ...,
#     'starting_compound': 'soft',
#     'used_tyres': [('soft', 20), ('medium', 25)],
#     'current_tyre': 'hard'
# }

# Creating a dictionary of driver objects that will house all their information

drivers = {driver: Driver(driver,
                          drivers_tyre_models[driver],
                          drivers_consistency[driver],
                          number_of_stops=3,
                          top_n_strategies_parameter=1,
                          alternate_strategies_method='breadth',
                          start_compound=race_state[driver]['current_tyre'],
                          race_configuration=race_config,
                          )

           for driver in race_state}

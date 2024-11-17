"""Correlation data to see the accuracy of current simulation logic - models for Abu-Dhabi 2017, extrapolated from
TUM"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sympy import symbols, Piecewise, lambdify

from RaceStrategyEngine.driver import Driver
from RaceStrategyEngine.race_configuration import RaceConfiguration
from RaceStrategyEngine.tyre_model import TyreModel

# Define the symbol x
x = symbols('x')

driver_models_new = {
    'Verstappen': {
        "ultra_soft": TyreModel(100.144 + 0.018 * x),
        "super_soft": TyreModel((100.144 + 0.127) + 0.036 * x),
        # "hard": TyreModel((100.144 + 0.427) + 0.046 * x),  # Not from data don't use
    },
    'Ricciardo': {
        "ultra_soft": TyreModel(100.144 + 0.050 * x),
        "super_soft": TyreModel((100.144 + 0.556) + 0.050 * x),
        # "hard": TyreModel((100.144 + 0.856) + 0.060 * x),  # Not from data don't use
    },
    'Hamilton': {
        "ultra_soft": TyreModel(99.9 + 0.020 * x),
        "super_soft": TyreModel((99.9 + 0.900) + 0.030 * x),
        # "hard": TyreModel((99.9 + 1.2) + 0.04 * x),  # Not from data don't use

    },
    'Bottas': {
        "ultra_soft": TyreModel(99.9 + 0.020 * x),
        "super_soft": TyreModel((99.9 + 0.600) + 0.020 * x),
        # "hard": TyreModel((99.9 + 0.900) + 0.030 * x),  # Not from data don't use
    },
    'Vettel': {
        "ultra_soft": TyreModel(100.144 + 0.020 * x),
        "super_soft": TyreModel((100.144 + 0.868) + 0.013 * x),
        # "hard": TyreModel((100.144 + 1.286) + 0.023 * x),  # Not from data don't use
    },
    'Raikkonen': {
        "ultra_soft": TyreModel(100.144 + 0.029 * x),
        "super_soft": TyreModel((100.144 + 0.290) + 0.039 * x),
        # "hard": TyreModel((100.144 + 0.590) + 0.049 * x),  # Not from data don't use

    },
}

correlation_race_state = {
    'Ricciardo': {'position': 4, 'total_time': 0, 'lap_time': 0, 'delta_to_leader': 0, 'used_tyres': [], 'current_tyre': 'ultra_soft', 'current_tyre_laps_age': 0},
    'Raikkonen': {'position': 5, 'total_time': 0, 'lap_time': 0, 'delta_to_leader': 0, 'used_tyres': [], 'current_tyre': 'ultra_soft', 'current_tyre_laps_age': 0},
    'Hamilton': {'position': 2, 'total_time': 0, 'lap_time': 0, 'delta_to_leader': 0, 'used_tyres': [], 'current_tyre': 'ultra_soft', 'current_tyre_laps_age': 0},
    'Vettel': {'position': 3, 'total_time': 0, 'lap_time': 0, 'delta_to_leader': 0, 'used_tyres': [], 'current_tyre': 'ultra_soft', 'current_tyre_laps_age': 0},
    'Verstappen': {'position': 6, 'total_time': 0, 'lap_time': 0, 'delta_to_leader': 0, 'used_tyres': [], 'current_tyre': 'ultra_soft', 'current_tyre_laps_age': 0},
    'Bottas': {'position': 1, 'total_time': 0, 'lap_time': 0, 'delta_to_leader': 0, 'used_tyres': [], 'current_tyre': 'ultra_soft', 'current_tyre_laps_age': 0},
}

driver_strategies_correlation = {
    "Verstappen": (58.0, ("ultra_soft", 14), ("super_soft", 41)),
    "Ricciardo": (58.0, ("ultra_soft", 19), ("super_soft", 36)),
    "Hamilton": (58.0, ("ultra_soft", 24), ("super_soft", 31)),
    "Bottas": (58.0, ("ultra_soft", 21), ("super_soft", 34)),
    "Vettel": (58.0, ("ultra_soft", 20), ("super_soft", 35)),
    "Raikkonen": (58.0, ("ultra_soft", 15), ("super_soft", 40)),
}

drivers_data_consistency = {
    "Verstappen": {"mean": 0.637, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 3.0,
                   "std_dev_tyre_change_time": 0.1},
    "Ricciardo": {"mean": 0.0, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 3.0,
                  "std_dev_tyre_change_time": 0.1},
    "Hamilton": {"mean": 0.0, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 3.0,
                 "std_dev_tyre_change_time": 0.1},
    "Bottas": {"mean": 0.1, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 3.0,
               "std_dev_tyre_change_time": 0.1},
    "Vettel": {"mean": 0.0, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 3.0,
               "std_dev_tyre_change_time": 0.1},
    "Raikkonen": {"mean": 0.307, "std_dev": 0.1, "min_lap_time_variation": 0.05, "mean_tyre_change_time": 3.0,
                  "std_dev_tyre_change_time": 0.1},
}

# Define the new overtake chances function
overtake_chances_piecewise = Piecewise(
    (0, x < 1),
    (1, True)
)
overtake_chances_metric = lambdify(x, overtake_chances_piecewise, modules=[{"Piecewise": np.piecewise}, "numpy"])

# Create a new instance of RaceConfig with the provided parameters
correlation_race_params = RaceConfiguration(
    points_distribution={1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1},
    pit_lane_time_loss=20.5,
    # full_course_yellow_pitloss=None,
    probability_of_full_course_yellow=None,
    num_laps=55,
    total_fuel=100,
    fuel_effect_seconds_per_kg=0.033,
    fuel_consumption_per_lap=1.79,
    drs_boost=0.8,
    drs_activation_lap=3,
    time_lost_due_to_being_overtaken=0.6,
    time_lost_performing_overtake=0.1,
    min_time_lost_due_to_failed_overtake_attempt=0.1,
    max_time_lost_due_to_failed_overtake_attempt=0.6,
    overtake_chances_piecewise=overtake_chances_piecewise,
    race_start_stationary_time_penalty=2.5,
    race_start_grid_position_time_penalty=1.0,
    track_evolution=None
)

correlation_drivers = {
    driver: Driver(driver, driver_models_new[driver], drivers_data_consistency[driver], number_of_stops=3,
                   race_configuration=correlation_race_params) for driver in driver_models_new}

# just setting the selected strategy the one i want (TUM) not the optimastion result
for driver in correlation_drivers:
    correlation_drivers[driver]._selected_strategy = driver_strategies_correlation[driver]

# # For backward compatibility and ease of use in existing code
# correlation_race_params = correlation_race_params.to_dict()

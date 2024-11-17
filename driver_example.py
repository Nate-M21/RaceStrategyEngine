from sympy import symbols, Piecewise
from RaceStrategyEngine.driver import Driver
from RaceStrategyEngine.tyre_model import TyreModel
from simulation_parameters_example import race_config

x = symbols('x')

# Define tyre models
# Each tyre compound can have be a different type of equation at different stages to model tyre complexity
# This example uses linear models for all compounds at all stages
example_tyres = {
    "hyper_soft": TyreModel(89.0 + 0.15 * x),
    "ultra_soft": TyreModel(89.2 + 0.12 * x),
    "super_soft": TyreModel(89.32 + 0.11 * x), 
    "soft": TyreModel(89.44 + 0.1 * x),
    "medium": TyreModel(89.7 + 0.06 * x),
    "hard": TyreModel(90.0 + 0.03 * x),
    "super_hard": TyreModel(90.3 + 0.015 * x),
}

# IMPORTANT: Use only SymPy (sp) functions and expressions for the TyreModel instance. Do NOT use math.log, np.exp,
# or similar.
# Examples of SymPy functions to use:
# - sp.exp(x): exponential function
# - sp.log(x): natural logarithm
# - sp.sqrt(x): square root
# - sp.sin(x), sp.cos(x): trigonometric functions
# - sp.Max(x, y), sp.Min(x, y): maximum and minimum functions
# etc
# All expressions must be instances of sp.Expr

# Example with only 3 tyre choices for a race, using piecewise functions for more complex modeling
# Each part of the tyre model can be a different type of equation since F1 tyres are complex.
# You can have a warm-up phase where tyres are initially slower, then a linear performance phase with gradual increase in performance and degradation,
# and finally an exponential 'cliff' phase where there is substantial drop-off if desired.
# For simplicity, all phases are linear but with varying degrees of severity based on laps run.

race_tyres = {
    "soft": TyreModel(Piecewise(
        (89.44 + 0.1 * x, x < 10),  # Initial performance, high degradation
        (94.44 + 0.2 * (x - 10), (x >= 10) & (x < 20)),  # Increased degradation after 10 laps
        (96.44 + 0.5 * (x - 20), x >= 20)  # Severe degradation after 20 laps
    )),
    "medium": TyreModel(Piecewise(
        (89.7 + 0.06 * x, x < 15),  # Lower initial degradation, lasts longer
        (90.6 + 0.12 * (x - 15), (x >= 15) & (x < 25)),  # Slight increase in degradation
        (92.8 + 0.3 * (x - 25), x >= 25)  # Steep drop-off after 25 laps
    )),
    "hard": TyreModel(Piecewise(
        (90 + 0.03 * x, x < 20),  # Least degradation initially, for longest-lasting performance
        (90.6 + 0.06 * (x - 20), (x >= 20) & (x < 30)),  # Mild degradation increase
        (91.2 + 0.1 * (x - 30), x >= 30)  # Gradual degradation after 30 laps
    ))
}

# Initialize a test driver
# The Driver class requires a RaceConfig instance to ensure consistency across all drivers in a race simulation
test_driver_consistency = {
    'mean': 0.15,  # Average deviation from ideal lap time in seconds
    'std_dev': 0.1,  # Standard deviation for lap time variability in seconds
    'min_lap_time_variation': 0.05,  # Minimum variation applied to each lap time in seconds
    'mean_tyre_change_time': 2.3,  # Average time taken to change tyres during a pit stop in seconds
    'std_dev_tyre_change_time': 0.2  # Standard deviation of tyre change time in seconds
}

test_driver = Driver('test', race_tyres, test_driver_consistency, number_of_stops=3, race_configuration=race_config)

# Alternative: Use StrategyCombinatorics directly for more flexibility in parameters
# test_driver = StrategyCombinatorics(tyre_information=race_tyres, number_of_stops=5, num_laps=55, pit_loss=24)

# Display the dataframe of optimized 'clean-air' strategies
print(test_driver.strategy_options)

print('\nSelecting best strategies'.title(),'\n','-'*100)
# View the distribution of laps for the best strategy in the whole dataframe
print(f"Fastest Strategy: {test_driver.strategy.best_strategy(dist_mode=True)}", end='\n\n')

# View from one stop onwards too # Numerical (int) between the underscore is also allowed - test_driver.strategy.best_n_stop
# The limit you have is the argument you set for the 'number_of_stops' parameter, in this example it would be 3
print(f"Fastest One Stop: {test_driver.strategy.best_one_stop}") #  alternative - test_driver.strategy.best_1_stop
print(f"Fastest Two Stop: {test_driver.strategy.best_two_stop}") #  alternative - test_driver.strategy.best_2_stop
print(f"Fastest Three Stop: {test_driver.strategy.best_three_stop}") #  alternative - test_driver.strategy.best_3_stop

# Plot the tyre performance models
test_driver.strategy.plot_tyre_performance()

# Visualize how different strategies perform in 'clean-air'
number_to_view = 5 
# The method below automatically selects the top n when you select a number to view, if you want to view slower strategies increase the number
test_driver.strategy.plot_strategy_performance(number_of_strategies=number_to_view)


# Additional parameters for advanced usage in the Monte Carlo simulations for the driver:

# top_n_strategies_parameter: Number of top strategies to consider for each pit stop level.
# Useful for exploring multiple strategy options in Monte Carlo simulations.
# Default is 1 (only the best strategy) utilizing the 'breadth' method - so best for each stop.

# stops_depth_limit: Maximum number of pit stops to consider when evaluating strategies.
# Helps focus analysis within realistic operational limits.
# Default is None (considers all stops up to number_of_stops).

# alternate_strategies_method: Determines how alternate strategies are calculated.
# Options: 'breadth' (strategies for each stop level), 'depth' (top N overall),
# or 'all' (all strategies up to stops_limit).
# Affects diversity of strategy options in Monte Carlo simulations.

# alternate_strategies_dict_mode: If True, returns alternate strategies as a dictionary;
# if False, returns as a list. Offers flexibility in data structure for analysis.

# start_compound: Specifies the starting tyre compound.
# Influences strategy select strategies starting with this compound. And prune combinations that don't have it in their combination
# Default is None (no preference).

# Example usage with all parameters:
test_driver = Driver('test', example_tyres, test_driver_consistency, number_of_stops=3, race_configuration=race_config,
                     top_n_strategies_parameter=5, stops_depth_limit=2,
                     alternate_strategies_method='depth',
                     start_compound='soft',
                     available_tyres_constraint=[('soft', 2), ('soft', 0), ('medium', 0), ('hard', 2)])

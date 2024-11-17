from RaceStrategyEngine.monte_carlo_race_simulations import MonteCarloRaceSimulations
from RaceStrategyEngine.monte_carlo_analysis import (plot_race_strategy_ranking,
                                                     plot_drivers_finishing_position_frequency,
                                                     plot_race_strategy_position_distribution,
                                                     plot_traffic_heatmap, plot_traffic_boxplot,
                                                     plot_drivers_position_dominance)
from simulation_parameters_example import drivers, race_state, race_config

# Initialize the Monte Carlo race simulation
test_race = MonteCarloRaceSimulations(drivers=drivers, starting_race_grid=race_state, race_configuration=race_config, lap_variation_range=(-10,10))

# Run a single Monte Carlo simulation with randomized strategies and pit stops timings
test_race.run_monte_carlo_simulation()

# Display the results of the simulation in a clean format
test_race.display()

# Get and print the detailed results of the Monte Carlo simulation (set format_json=False for a Python dictionary)
result = test_race.get_result_from_simulation_run(format_json=True, json_indent='\t')
print(result)

# You can utilize the following methods to inherited from the RaceStrategyEngine
# Get and print the race trace data for plotting
result = test_race.get_result_for_race_trace_plotting(format_json=True, json_indent=None)
# Uncomment the next line
# print(result)

# Plot the race trace for a quick overview of how the race progressed
# Uncomment the next line
# test_race.plot_race_trace()

# Plot the tyre usage to visualize each driver's tyre strategy throughout the race
# Uncomment the next line
test_race.plot_tyre_usage()

# Quick view of the starting grid
print('\nStrarting Grid')
print('-'*30)
for driver in sorted(race_state, key = lambda x: race_state[x]['position']):
    print(driver, f'starting position: P{race_state[driver]['position']}')
print('-'*30, end='\n\n')
# To view more detailed information on the changeable simulation parameters set for this race example,
# refer to RaceStrategyEngine/simulation_parameters_example.py

# Select a specific driver to run multiple simulations for (Any driver in the race state, i.e., simulation starting grid, can be chosen)
selected_driver = 'Leclerc'
selected_drivers = ['Leclerc', 'Hamilton', 'Verstappen']
# Run simulations using multi-core processing (requires Java and PySpark)
df = test_race.run_and_analyze_simulations_from_monte_carlo_runs(drivers_to_analyse='ALL',
                                                                 num_simulations=10_000,
                                                                 method='multi-core',
                                                                 immediate_plot=False)

# If you run into issues with the multi-core or want to try single-core processing, uncomment the following line
# (and then comment out the multi-core line above)
# df = test_race.run_and_analyze_simulations_from_monte_carlo_runs(driver_to_analyse=selected_driver, num_simulations=1_000, method='single-core', immediate_plot=False)

# Display the results dataframe
print(df)

# # Generate various plots to visualize the simulation results

# TODO need description, also make one colour scale that all drivers share like my 3D scatter plot
plot_race_strategy_ranking(pandas_df=df)

# # Parallel coordinates plot: Shows relationships between pit stop timings and points (haven't added ML
# interpolation yet)
test_race.plot_parallel_coordinate_plot(pandas_df=df)

# Scatter plot with quadratic fit: Illustrates which strategy will often perform (score) better in a race situation,
# not just alone as in a 'free-air' model. Additionally, what first pit stop lap on average tends to maximize the
# points scored for that strategy.
test_race.plot_race_strategy_performance(pandas_df=df)

# Bar plot: Displays the frequency of each finishing position
plot_drivers_finishing_position_frequency(pandas_df=df)

# # Histogram: Shows the distribution of points outcomes for each strategy
plot_race_strategy_position_distribution(pandas_df=df)

# Basic 3D scatter plot: Visualizes relationships between pit stop laps and points
test_race.plot_three_dimensional_scatter_plot_basic(pandas_df=df)
#
# 3D scatter plot with jitter: Helps visualize overlapping points
# Note: This adds columns to the dataframe. Use a deep copy if you want to preserve the original
test_race.plot_three_dimensional_scatter_plot(pandas_df=df)

# Heatmap: Shows mean laps behind traffic for different strategies
plot_traffic_heatmap(pandas_df=df)

# Box plot: Displays the distribution of laps behind traffic for each strategy
plot_traffic_boxplot(pandas_df=df)

# TODO need description
plot_drivers_position_dominance(pandas_df=df)

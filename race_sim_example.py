from RaceStrategyEngine.race_simulation import RaceSimulation
from simulation_parameters_example import drivers, race_state, race_config

# Initialize the race simulation
test_race = RaceSimulation(drivers=drivers, starting_race_grid=race_state, race_configuration=race_config)

# To create your own race_configuration / track characteristics you would use RaceConfiguration instance 
# in RaceStrategyEngine/race_configuration.py. To view more information on the customizable parameters for a
# simulation see the parameters file. They are in directory RaceStrategyEngine/simulation_parameters_example.py

# View the current race configuration / track characteristics
print(race_config)

# Run the simulation (required before accessing results or plots)
test_race.run_simulation()

# Display the simulation results in a clean format
test_race.display()

# Plot the tyre usage to visualize each driver's tyre strategy throughout the race
test_race.plot_tyre_usage()

# Plot the race trace for a quick overview of how the race progressed
test_race.plot_race_trace()

# Get a summary of the race results as a JSON string (set format_json=False for a Python dictionary)
result = test_race.get_result_from_simulation_run(format_json=True, json_indent=4)
# print(result)

# Get a summary of the race gaps (JSON format)
gap_result = test_race.get_result_for_race_trace_plotting(format_json=True, json_indent=None)
# Uncomment the next line to view the gap results
# print(gap_result)


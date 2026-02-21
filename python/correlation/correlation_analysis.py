import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RaceStrategyEngine.race_simulation import RaceSimulation
from simulation_parameters_example import drivers, race_state, race_config, drivers_consistency

# Initialize the race simulation

from correlation_simulation_parameters import correlation_drivers, correlation_race_state, correlation_race_config, drivers_data_consistency
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import rust_sim_core

corl_race = RaceSimulation(correlation_drivers, correlation_race_state, correlation_race_config)
drivers_consistency = drivers_data_consistency
race_config = correlation_race_config

# corl_race = RaceSimulation(drivers=drivers, starting_race_grid=race_state, race_configuration=race_config)

num_of_sims = 10_000

simulation_results = defaultdict(list)
rust_simulation_results = defaultdict(list)

start = time.perf_counter()
for _ in tqdm(range(num_of_sims)):
    corl_race.run_simulation()
    result = corl_race.get_result_from_simulation_run(format_json=False)
    # data = corl_race.rust_info
    # rust_result = rust_sim.lap_discrete_simulation(data)
    for driver, info in result.items():
        simulation_results[f"{driver}_time"].append(info['race_time'])
        simulation_results[f"{driver}_position"].append(info['end_position'])

    # for driver, info in rust_result.items():
    #     rust_simulation_results[f"{driver}_time"].append(info['total_time'])
    #     rust_simulation_results[f"{driver}_position"].append(info['position'])


end = time.perf_counter()
print(f"Python duration is {end - start}")

# print(simulation_results)

simulation_results = pd.DataFrame(simulation_results)
# print(simulation_results)



avg_times = simulation_results.filter(like='_time').mean()
avg_positions = simulation_results.filter(like='_position').mean()

print(f"Average after {num_of_sims} runs".title())
print('-' * 100)
print(avg_times)
print('\n')
print(avg_positions, end='\n\n')

print('frequency of positions achieved'.title())
print('-' * 100)
position_counts = simulation_results.filter(like='_position').apply(pd.Series.value_counts).fillna('None')
print(position_counts, end='\n\n')

bottas_beat_hamilton = simulation_results['Hamilton_position'] > simulation_results['Bottas_position']

races_bottas_won = simulation_results[bottas_beat_hamilton]


races_bottas_won_avg_times = races_bottas_won.filter(like='_time').mean()
races_bottas_won_avg_positions = races_bottas_won.filter(like='_position').mean()

print('#' * 100)
print('average results for races when Bottas won'.title())
print('-' * 100)
print(races_bottas_won_avg_times)
print('\n')
print(races_bottas_won_avg_positions)



print("#"*100)
print("Rust Version")
simulation_results = defaultdict(list)
# num_of_sims = 100_000



# Extract only the parameters that your Rust Simulation constructor accepts


# Create the simulation

corl_race._enable_exp_rust_backend()
start = time.perf_counter()
for _ in tqdm(range(num_of_sims)):
    result = corl_race._exp_run_simulation(False)
    for driver, info in result.items():
        simulation_results[f"{driver}_time"].append(info['total_time'])
        simulation_results[f"{driver}_position"].append(info["position"])

end = time.perf_counter()
print(f"Rust duration is {end - start}")

simulation_results = pd.DataFrame(simulation_results)



avg_times = simulation_results.filter(like='_time').mean()
avg_positions = simulation_results.filter(like='_position').mean()

print(f"Average after {num_of_sims} runs".title())
print('-' * 100)
print(avg_times)
print('\n')
print(avg_positions, end='\n\n')

print('frequency of positions achieved'.title())
print('-' * 100)
position_counts = simulation_results.filter(like='_position').apply(pd.Series.value_counts).fillna('None')
print(position_counts, end='\n\n')

bottas_beat_hamilton = simulation_results['Hamilton_position'] > simulation_results['Bottas_position']

races_bottas_won = simulation_results[bottas_beat_hamilton]


races_bottas_won_avg_times = races_bottas_won.filter(like='_time').mean()
races_bottas_won_avg_positions = races_bottas_won.filter(like='_position').mean()

print('#' * 100)
print('average results for races when Bottas won'.title())
print('-' * 100)
print(races_bottas_won_avg_times)
print('\n')
print(races_bottas_won_avg_positions)

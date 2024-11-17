import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RaceStrategyEngine.race_simulation import RaceSimulation
from correlation_simulation_parameters import correlation_drivers, correlation_race_state, correlation_race_params
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

corl_race = RaceSimulation(correlation_drivers, correlation_race_state, correlation_race_params)

num_of_sims = 10000

simulation_results = defaultdict(list)

for _ in tqdm(range(num_of_sims)):
    corl_race.run_simulation()
    result = corl_race.get_result_from_simulation_run(format_json=False)
    for driver, info in result.items():
        simulation_results[f"{driver}_time"].append(info['race_time'])
        simulation_results[f"{driver}_position"].append(info['end_position'])

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

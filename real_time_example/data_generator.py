import json
import sys
import time
from pathlib import Path

from pathlib import Path
import json

script_dir = Path(__file__).parent

path1 = script_dir / 'driver_laptimes.json'
lap_time_content = path1.read_text()
driver_laptimes = json.loads(lap_time_content)

path2 = script_dir / 'list_of_race_states.json'
race_state_content = path2.read_text()
list_of_race_states = json.loads(race_state_content)

# Initial race state for lap 1
initial_race_state = {
    'Leclerc': {'position': 2, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                'current_tyre_laps_age': 0},
    'Hamilton': {'position': 4, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                 'current_tyre_laps_age': 0},
    'Verstappen': {'position': 1, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': 'medium',
                   'current_tyre_laps_age': 0},
    'Gasly': {'position': 5, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
              'current_tyre_laps_age': 0},
    'Norris': {'position': 3, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
               'current_tyre_laps_age': 0},
    'Ocon': {'position': 6, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
             'current_tyre_laps_age': 0},
    'Piastri': {'position': 7, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                'current_tyre_laps_age': 0},
    'Perez': {'position': 8, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
              'current_tyre_laps_age': 0},
    'Sainz': {'position': 9, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
              'current_tyre_laps_age': 0},
    'Russell': {'position': 10, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                'current_tyre_laps_age': 0},
    "Alonso": {'position': 11, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
               'current_tyre_laps_age': 0},
    "Stroll": {'position': 12, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
               'current_tyre_laps_age': 0},
    "Sargeant": {'position': 13, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                 'current_tyre_laps_age': 0},
    "Albon": {'position': 14, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
              'current_tyre_laps_age': 0},
    "Zhou": {'position': 15, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
             'current_tyre_laps_age': 0},
    "Bottas": {'position': 16, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
               'current_tyre_laps_age': 0},
    "Magnussen": {'position': 17, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                  'current_tyre_laps_age': 0},
    "Hulkenberg": {'position': 18, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                   'current_tyre_laps_age': 0},
    "Ricciardo": {'position': 19, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                  'current_tyre_laps_age': 0},
    "Tsunoda": {'position': 20, 'total_time': 0, "delta_to_leader": 0, 'used_tyres': [], 'current_tyre': None,
                'current_tyre_laps_age': 0},
}

laps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
indices = [0,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]




lap_to_index ={}
for index,lap in enumerate(laps):
    lap_to_index[lap] = indices[index]

def generate_race_data():
    current_lap = 1
    max_lap = 55


    while True:
        index_to_use = lap_to_index[current_lap]
        laptimes_to_use = driver_laptimes

        race_state_to_use: dict = list_of_race_states[index_to_use]
        
        yield {
            'laptimes': laptimes_to_use,
            'race_state': race_state_to_use,
            'current_lap': current_lap
        }
        


        current_lap += 1
        if current_lap > max_lap:
            current_lap = 1
        
        time.sleep(5) # 60

# Create a single instance of the generator
data_generator = generate_race_data()

def race_data_generator():
    return next(data_generator)

if __name__ == '__main__':
    # while True:
    #     a = race_data_generator()
    #     break
    #     # print(a)
    lap = 17
    index_to_use = lap_to_index[lap]
    print(list_of_race_states[index_to_use]['Alonso'])



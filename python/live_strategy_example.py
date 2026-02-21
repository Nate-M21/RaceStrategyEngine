import sys
from RaceStrategyEngine.live_strategy import LiveStrategy
import threading
from real_time_example.data_generator import race_data_generator
from simulation_parameters_example import drivers, race_state, race_config
from RaceStrategyEngine.utility import RaceDataPacket

shared_dict = {
        'current_lap': 1,
        'race_state': race_state,
        'laptimes': {driver: [] for driver in drivers}
    }



data_lock = threading.Lock()

def daemon_generation_thread(shared_dict):
    while True:
        new_data = race_data_generator()

        # print(new_data)

        with data_lock:
            shared_dict.update(new_data)





# # Start the data generation thread
data_thread = threading.Thread(target=daemon_generation_thread, args=(shared_dict,))
data_thread.daemon = True
data_thread.start()


engine = LiveStrategy(drivers, race_state, race_config)

engine.start_engine()

try:
    while True:

        with data_lock:
            current_lap = shared_dict['current_lap']
            race_state_to_use = shared_dict['race_state']
            laptimes_to_use = shared_dict['laptimes']

        sys.stdout.write(f"\rLap: {current_lap}  ")
        sys.stdout.flush()
        # print(f"Current lap in python: {current_lap}")

        new_data = RaceDataPacket(current_lap=current_lap, race_state=race_state_to_use, laptimes=laptimes_to_use)

        engine.run_engine(new_data, gap_delta_method=False)

        result = engine.get_predictions()

        for driver, strategy in result.items():
             engine.update_driver_race_strategy(driver, strategy)

        engine.run_simulation()

        predicted_gaps = engine.get_result_for_race_trace_plotting(format_json=False)

        # print(predicted_gaps)

        predicted_gaps['current_lap'] = current_lap

        engine.send_zmq_message(predicted_gaps)

except KeyboardInterrupt:
        print('\n\nSo box box.\nBox box. \nBox this lap. \nPit confirm, need to you come in mate, easy in.\n')

finally:
    engine.stop_engine()
    print("Engine stopped.")




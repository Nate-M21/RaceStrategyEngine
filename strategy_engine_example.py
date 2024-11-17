# strategy_engine_example.py
import time
import threading
import multiprocessing
from RaceStrategyEngine.race_strategy_engine import RaceStrategyEngine
from RaceStrategyEngine.utility import RaceDataPacket
from simulation_parameters_example import drivers, race_state, race_config, strategy_weighting
from real_time_example.data_generator import race_data_generator
import sys


def data_generation_thread(shared_dict, race_is_ongoing):
    while race_is_ongoing[0]:
        new_data = race_data_generator()
        shared_dict.update(new_data)
        time.sleep(0.001)


def main():
    # Create a multiprocessing.Manager instance
    manager = multiprocessing.Manager()

    # Initialize the race simulation engine with the manager
    engine = RaceStrategyEngine(
        drivers=drivers,
        starting_race_grid=race_state,
        race_configuration=race_config,
        race_strategy_weighting=strategy_weighting,
        num_worker_processes=2,
        num_redis_processes=3,
        gap_delta_method=False,


    )

    shared_dict = manager.dict()
    shared_dict.update({
        'current_lap': 1,
        'race_state': race_state,
        'laptimes': {driver: [] for driver in drivers}
    })

    # Start the data generation thread
    race_is_ongoing = [True]
    data_thread = threading.Thread(target=data_generation_thread, args=(shared_dict, race_is_ongoing))
    data_thread.daemon = True
    data_thread.start()

    try:
        # Start the strategy engine processes
        engine.start_strategy_engine()

        while True:
            current_lap = shared_dict['current_lap']
            race_state_to_use = shared_dict['race_state']
            laptimes_to_use = shared_dict['laptimes']
            new_data = RaceDataPacket(current_lap=current_lap, race_state=race_state_to_use, laptimes=laptimes_to_use)

            # Ingest the latest data into the simulation
            engine.ingest_latest_data(new_data=new_data)

            engine.run_strategy_engine()

            predicted_result, predicted_positions, predicted_gaps = engine.get_simulation_results()
            predicted_gaps['current_lap'] = current_lap
            sys.stdout.write(f"\rLap: {current_lap}  ")
            sys.stdout.flush()
            engine.send_zmq_message(predicted_gaps)

    except KeyboardInterrupt:
        print('\n\nStop the car!\nStop the car! \nWe have a problem. \nSorry mate, need to retire, stop the car.\n')

    finally:
        engine.stop_strategy_engine()
        race_is_ongoing[0] = False
        data_thread.join()
        print("Engine stopped.")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()

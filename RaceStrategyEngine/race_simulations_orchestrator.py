"""
This module implements a real-time, multicore Monte Carlo simulation system for race strategy optimization.

It orchestrates the execution of concurrent Monte Carlo simulations across multiple CPU cores,
providing probabilistic reasoning about the best strategies for all drivers. The system continuously
adapts to the live race state, allowing all cores to work with the latest data.

Results from all cores are aggregated and can be used for further analysis or integration with other systems.
This module serves as the backbone for race strategy optimization in a larger race simulation system.

Note: This module requires a minimum of 8 CPU cores for efficient operation.
"""
from collections import namedtuple
import multiprocessing
import time
import dill
from typing import Literal
import logging
from .monte_carlo_race_simulations import MonteCarloRaceSimulations
from .utility import RaceDataPacket, SharedRedisResultStack, SharedRaceStateStore, RedisConnectionParameters
from .driver import Driver
from .race_configuration import RaceConfiguration

SerializedSimulationObjects = namedtuple('SerializedSimulationObjects', ['serialized_drivers',
                                                                         'serialized_race_state',
                                                                         'serialized_race_config',
                                                                         'serialized_schema'])




def setup_logging():
    logging.basicConfig(filename='race_strategy_predictor.log',
                        level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d - %(processName)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    return logging.getLogger(__name__)


class RaceSimulationsOrchestrator:
    """Orchestrates multiple parallel monte carlo race simulations.

    This class manages the execution of concurrent Monte Carlo simulations across multiple CPU cores,
    coordinating data flow, state updates, and result processing. It serves as the central hub for
    the race simulation system, handling initialization, process management, data distribution,
    and result collection.

    Attributes:
        num_worker_processes (int): Number of worker processes to spawn.




    """

    def __init__(self, drivers: dict[str, Driver], starting_grid: dict, race_configuration: RaceConfiguration,
                 num_worker_processes: int = 1, gap_delta_method=True,
                 redis_host='localhost', redis_port=6379, redis_db=0, shared_state_db=1,
                 output_format: Literal['JSON', 'PYTHON'] = 'PYTHON'):

        initial_laptimes = {driver: [] for driver in starting_grid}

        initial_data = RaceDataPacket(laptimes=initial_laptimes, race_state=starting_grid, current_lap=1)

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.shared_state_db = shared_state_db

        # I am creating a temporary instance to get the schema, that all workers should have
        # As the schema is dependent on arguments, all workers will have different instance but same
        # inputs, so they will have the same schema and outputting the same structure
        self.schema = MonteCarloRaceSimulations(drivers, starting_grid, race_configuration).schema

        # I'm using dill because normal serialization fails
        self.serialized_drivers = dill.dumps(drivers)
        self.serialized_race_state = dill.dumps(starting_grid)
        self.serialized_race_config = dill.dumps(race_configuration)
        self.serialized_schema = dill.dumps(self.schema)

        self.redis_shared_race_state = SharedRaceStateStore(host=self.redis_host, port=self.redis_port,
                                                            db=self.shared_state_db)

        self.redis_shared_race_state.update_shared_race_state(new_race_data=initial_data)

        self.serialized_data = SerializedSimulationObjects(serialized_drivers=self.serialized_drivers,
                                                           serialized_race_state=self.serialized_race_state,
                                                           serialized_race_config=self.serialized_race_config,
                                                           serialized_schema=self.serialized_schema)

        self.redis_params = RedisConnectionParameters(redis_host=self.redis_host, redis_port=self.redis_port,
                                                      redis_db=self.redis_db,
                                                      shared_state_db=self.shared_state_db)

        self.gap_delta_method = gap_delta_method

        if output_format not in ['JSON', 'PYTHON']:
            raise ValueError("The format of the out can only be JSON or PYTHON, which is list "
                             "of Spark Row objects, a json string and python dict respectively")

        self.output_format = output_format

        cpu_cores = multiprocessing.cpu_count()

        
        if cpu_cores < 8:
            raise SystemError("This program requires at least 8 CPU cores to run efficiently.")

        # under commiting so easy on the computer
        self.num_worker_processes = num_worker_processes

        self._processes = []

    def start_orchestrator(self):

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"Starting {self.num_worker_processes} simulation worker processes...")

        for _ in range(self.num_worker_processes):
            worker_process = multiprocessing.Process(
                target=RaceSimulationsOrchestrator._simulation_worker,
                kwargs={
                    'serialized_simulation_parameters': self.serialized_data,
                    'gap_delta_method': self.gap_delta_method,
                    'redis_parameters': self.redis_params,
                    'output_format': self.output_format
                }

            )
            self._processes.append(worker_process)

        for process in self._processes:
            process.start()

        print(f"Started at {current_time}")

    def update_data(self, new_data: RaceDataPacket):
        if not isinstance(new_data, RaceDataPacket):
            self.stop_orchestrator()
            raise TypeError(f'Expected RaceDataPacket, got {type(new_data).__name__}')
        self.redis_shared_race_state.update_shared_race_state(new_race_data=new_data)

    @staticmethod
    def _simulation_worker(serialized_simulation_parameters: SerializedSimulationObjects,
                           redis_parameters: RedisConnectionParameters, gap_delta_method, output_format):

        serialized_drivers = serialized_simulation_parameters.serialized_drivers
        serialized_race_state = serialized_simulation_parameters.serialized_race_state
        serialized_race_config = serialized_simulation_parameters.serialized_race_config
        serialized_schema = serialized_simulation_parameters.serialized_schema

        redis_host = redis_parameters.redis_host
        redis_port = redis_parameters.redis_port
        shared_state_db = redis_parameters.shared_state_db

        redis_result_stack = SharedRedisResultStack(host=redis_host, port=redis_port, db=shared_state_db)

        shared_race_state = SharedRaceStateStore(host=redis_host, port=redis_port, db=shared_state_db)

        
        drivers = dill.loads(serialized_drivers)
        race_state = dill.loads(serialized_race_state)
        race_config = dill.loads(serialized_race_config)
        schema = dill.loads(serialized_schema)

        
        monte_race = MonteCarloRaceSimulations(drivers, race_state, race_config)
        assert monte_race.schema == schema, "Schema mismatch in worker process"  # making sure the schema is the same
        logger = setup_logging()

        while True:
            current_data = shared_race_state.get_shared_race_state()

            if current_data:
                monte_race.live_state_updates(live_race_data=current_data, gap_delta_method=gap_delta_method)
                logger.info(f"The current lap of the data being used in the"
                            f" simulation worker is lap {current_data.current_lap}")

                spark_result = monte_race.run_adaptive_monte_carlo_simulation()  # ROW OBJECT SPARK, not going
                # to use anymore
                result = spark_result

                if output_format == 'JSON':
                    result = monte_race.get_result_from_simulation_run(format_json=True)
                else:
                    result = monte_race.get_result_from_simulation_run(format_json=False)

                redis_result_stack.append(result)
                time.sleep(0.000001)  # Small sleep to prevent CPU workin too much
            else:
                time.sleep(0.0001)  # Small sleep to allow redis to update the shared state if it is empty

    def stop_orchestrator(self):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)

        for process in self._processes:
            process.terminate()
        for process in self._processes:
            process.join()
        print(f"Ended at {current_time}")
        print("All processes have been terminated.")

"""
This module implements a real-time race strategy prediction.

It utilizes Monte Carlo simulations across multiple cores to provide probabilistic reasoning
about the best strategies for all drivers. The system continuously adapts to the live race state,
allowing for dynamic strategy updates as the race progresses.

The module integrates with Redis for efficient storage and retrieval of simulation results,
and uses multiprocessing for parallel execution of simulations. It works in conjunction with
the RaceSimulationsOrchestrator to manage the actual running of simulations.

"""
import multiprocessing
import time
from redis import Redis
from .race_configuration import RaceConfiguration, RaceStrategyWeighting
from .race_simulations_orchestrator import RaceSimulationsOrchestrator
from .utility import RaceDataPacket, connect_to_redis, SharedRedisResultStack
import ast
import logging





def setup_logging():
    logging.basicConfig(filename='race_strategy_predictor.log',
                        level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d - %(processName)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    return logging.getLogger(__name__)


class RaceStrategyPredictor:
    def __init__(self, drivers, starting_grid, race_configuration: RaceConfiguration, gap_delta_method=True,
                 race_strategy_weighting: RaceStrategyWeighting | None = None,
                 num_worker_processes=1, num_redis_processes=2,
                 redis_host='localhost', redis_port=6379, redis_db=0, shared_state_db=1):

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.shared_state_db = shared_state_db
        self.race_strategy_weighting = race_strategy_weighting
        print('The race strategy weighting is', self.race_strategy_weighting)

        self.simulation_orchestrator = RaceSimulationsOrchestrator(drivers, starting_grid, race_configuration,
                                                                   num_worker_processes=num_worker_processes,
                                                                   redis_host=redis_host, redis_port=redis_port,
                                                                   redis_db=redis_db,
                                                                   shared_state_db=shared_state_db,
                                                                   gap_delta_method=gap_delta_method)
        self.num_redis_processes = num_redis_processes
        self._current_lap = 1

        self.schema = self.simulation_orchestrator.schema

        # Redis connection setup
        self.redis_client = None

        self._current_drivers = list(starting_grid.keys())

        self._redis_update_processes = []

        for _ in range(self.num_redis_processes):
            process = multiprocessing.Process(
                target=RaceStrategyPredictor.process_simulation_results,
                kwargs={
                    'redis_host': self.redis_host,
                    'redis_port': self.redis_port,
                    'redis_db': self.redis_db,
                    'shared_state_db': self.shared_state_db,
                    'race_strategy_weighting': self.race_strategy_weighting
                }
            )
            self._redis_update_processes.append(process)

    def start_predictor(self):
        """Start the simulation processes and result processing thread."""
        logger = setup_logging()
        logger.info("Starting predictor")

        self.redis_client = connect_to_redis(redis_host=self.redis_host, redis_port=self.redis_port,
                                             redis_db=self.redis_db)
        # Start all Redis update processes
        print(f"Starting {self.num_redis_processes} redis simulation processing processes...")

        for process in self._redis_update_processes:
            process.start()

        self.simulation_orchestrator.start_orchestrator()

    def stop_predictor(self):
        """Stop the simulation and result processing."""
        for process in self._redis_update_processes:
            process.terminate()

        self.simulation_orchestrator.stop_orchestrator()
        self.redis_client.flushall()  # Clear the Redis DB
        print(f"All Redis databases cleared. Address: {self.redis_host} | Port: {self.redis_port}")

    @staticmethod
    def process_simulation_results(redis_host, redis_port, redis_db, shared_state_db, race_strategy_weighting):
        """Continuously process results from the simulation orchestrator."""
        # Create a new Redis client in the child process
        redis_client = connect_to_redis(redis_host, redis_port, redis_db)
        result_stack = SharedRedisResultStack(host=redis_host, port=redis_port, db=shared_state_db)
        logger = setup_logging()
        # logger.info(f'The amount of data in the result stack is {len(result_stack)}')
        while True:
            if result_stack:
                result = result_stack.pop()
                logger.info(f'The amount of data in the result stack after popping {len(result_stack)}.')
                if result is not None:
                    RaceStrategyPredictor.store_simulation_result(redis_client, result, race_strategy_weighting)
            else:
                time.sleep(0.00001)  # I want to stop tight looping
                continue

    @staticmethod
    def store_simulation_result(redis_client: Redis, result, race_strategy_weighting: RaceStrategyWeighting | None):
        """Store a simulation result in Redis with TTL."""
        # Use a set to keep track of processed laps
        # Set up logging for this process
        logger = setup_logging()
        current_lap = None
        time_to_live = 90  # 60 * 5  # 5 minutes TTL, adjust as needed
        for driver, info in result.items():
            current_lap = info['simulation_start_lap']
            tyre_usage = info['tyre_usage']
            end_position = info['end_position']
            race_time = info['race_time']
            amount_of_stops = info['stops']

            lap_and_tyre_usage_key = f'{driver}:L{current_lap}:{tyre_usage}'

            occurence_count_key = 'occurence_count'
            end_position_key = 'sum_of_positions_achieved'
            race_time_key = 'sum_of_race_times_achieved'
            redis_client.hincrby(name=lap_and_tyre_usage_key, key=occurence_count_key, amount=1)
            redis_client.hincrby(name=lap_and_tyre_usage_key, key=end_position_key, amount=end_position)
            redis_client.hincrbyfloat(name=lap_and_tyre_usage_key, key=race_time_key, amount=race_time)

            total_count = int(redis_client.hget(name=lap_and_tyre_usage_key, key=occurence_count_key))
            end_position_sum = int(redis_client.hget(name=lap_and_tyre_usage_key, key=end_position_key))
            race_time_sum = float(redis_client.hget(name=lap_and_tyre_usage_key, key=race_time_key))

            mean_position = end_position_sum / total_count
            mean_race_time = race_time_sum / total_count

            strategy_score = mean_position

            if race_strategy_weighting:
                strategy_score = RaceStrategyPredictor.strategy_evaluation(race_strategy_weighting= race_strategy_weighting,
                                                                           mean_position=mean_position,
                                                                           race_time=mean_race_time,
                                                                           amount_of_stops=amount_of_stops)

            tyre_usage_key = f'{tyre_usage}'
            sorted_set_key = f'{driver}:L{current_lap}:strategy_scores'
            redis_client.zadd(sorted_set_key, {tyre_usage_key: strategy_score})

            # if sorted_set_key is accessed often it will live longer as it time will reset each time
            redis_client.expire(lap_and_tyre_usage_key, time_to_live)
            redis_client.expire(sorted_set_key, time_to_live)

        if current_lap is not None:
            # Signal that results for this lap are ready
            redis_client.set(f'lap_ready:{current_lap}', value='ready', ex=time_to_live)
            logger.info(f"Completed all drivers for lap {current_lap}")

    def get_predictions(self, wait_time=0.05):
        logger = setup_logging()
        predictions = {}
        current_lap = self._current_lap
        start_time = time.time()

        # Wait for results to be ready
        while not self.redis_client.exists(f'lap_ready:{current_lap}'):
            logger.warning(f"Waiting for results for lap {current_lap} to be ready...")
            elapsed_time = time.time() - start_time
            if elapsed_time > 5:
                logger.error(f'Waited over 5s to retrieve results for lap {current_lap}')
                logger.debug(f'lap {current_lap} current situation')
            time.sleep(wait_time)

        for driver in self._current_drivers:
            best_strategy = self.redis_client.zrange(f'{driver}:L{current_lap}:strategy_scores', 0, 0,
                                                     withscores=True)

            tyre_usage_bytes, _ = best_strategy[0]
            tyre_usage_str = tyre_usage_bytes.decode('utf-8')
            tyre_usage = ast.literal_eval(tyre_usage_str)
            predictions[driver] = {"tyre_usage": tyre_usage}

        return predictions, current_lap

    @staticmethod
    def strategy_evaluation(race_strategy_weighting: RaceStrategyWeighting, mean_position, race_time, amount_of_stops):
        """Give a strategy a score that will be used to compare against others"""

        position_weight = race_strategy_weighting.position_weighting
        race_time_weight = race_strategy_weighting.race_time_weighting
        stops_weight = race_strategy_weighting.amount_of_stops_weighting

        # Normalize race_time 
        normalized_time = race_time / race_strategy_weighting.estimated_total_race_time

        # Calculate score (lower is better)
        score = (position_weight * mean_position) + (race_time_weight * normalized_time) + (
                    stops_weight * amount_of_stops)

        return score

    def update_race_state(self, new_data: RaceDataPacket):
        """
        Update the race state in the simulation orchestrator.
        :param new_data: New race state data
        """
        logger = setup_logging()
        self._current_drivers = list(new_data.race_state.keys())
        self._current_lap = new_data.current_lap
        logger.info(f"The current lap of the data coming in is lap {self._current_lap}")
        self.simulation_orchestrator.update_data(new_data)

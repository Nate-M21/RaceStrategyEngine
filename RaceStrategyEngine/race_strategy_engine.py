"""
This module provides a high-level interface for race simulation and race strategy optimization.

The RaceStrategyEngine class encapsulates the complexity of race simulation and strategy prediction,
offering a streamlined interface for running simulations, applying predictions, and retrieving results.

Key features:
- Integration of race simulation and strategy prediction
- Real-time ingestion and processing of race data
- Application of predicted strategies to the main simulation
- Flexible result retrieval in various formats

This engine serves as the central component in a race strategy optimization system, managing the
interplay between the main race simulator and the strategy predictor. It's designed to be used
in real-time race analysis and strategy planning scenarios.

The engine utilizes multiprocessing for enhanced performance and can be configured with various
parameters such as the number of processes, Redis connection details, and gap delta calculation method.
"""
import os
import redis
from .race_simulation import RaceSimulation
from .race_strategy_predictor import RaceStrategyPredictor
from .utility import RaceDataPacket
from .driver import Driver
from .race_configuration import RaceConfiguration, RaceStrategyWeighting
from typing import Optional
import time
import zmq


class RaceStrategyEngine:

    def __init__(self, drivers: dict[str, Driver], starting_race_grid: dict, race_configuration: RaceConfiguration,
                 race_strategy_weighting: RaceStrategyWeighting | None = None,
                 gap_delta_method=True, num_worker_processes=1, num_redis_processes=1, address='*', port=5555,
                 redis_host='localhost', redis_port=6379, redis_db=0, shared_state_db=1) -> None:

        self._main_simulator = RaceSimulation(drivers=drivers,
                                              starting_race_grid=starting_race_grid,
                                              race_configuration=race_configuration)

        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_db = redis_db


        self._gap_delta_method = gap_delta_method

        self._socket = None
        self._address = address
        self._port = port
        # TODO Add the ability to change the protocol that zmq uses
        self._protocol = 'tcp'

        self._predictor_params = {
            'drivers': drivers,
            'starting_grid': starting_race_grid,
            'race_configuration': race_configuration,
            'race_strategy_weighting': race_strategy_weighting,
            'num_worker_processes': num_worker_processes,
            'num_redis_processes': num_redis_processes,
            'redis_host': redis_host,
            'redis_port': redis_port,
            'redis_db': redis_db,
            'shared_state_db': shared_state_db,
            'gap_delta_method': gap_delta_method
        }
        print('Engine', race_strategy_weighting)
        # I will be set up after Redis check, if successful avoiding downstream issues
        self._strategy_predictor = None

    def _setup_zmq_socket(self):
        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)

        try:
            self._socket.bind(f"tcp://{self._address}:{self._port}")
            print(f"ZeroMQ socket bound to tcp://{self._address}:{self._port}")

        except zmq.error.ZMQError as error:
            print(f"Unable to bind ZeroMQ socket. For address {self._address} Tried port {self._port}: {error}")
            os._exit(1)

    def _check_redis_server(self):
        try:
            redis_client = redis.Redis(host=self._redis_host, port=self._redis_port, db=self._redis_db)
            redis_client.ping()
        except redis.exceptions.ConnectionError:
            print("\nUnable to connect to Redis. Please ensure the Redis server is running.")
            os._exit(1)

    def start_strategy_engine(self):
        self._check_redis_server()
        self._setup_zmq_socket()

        print('Starting engine...')

        self._strategy_predictor = RaceStrategyPredictor(**self._predictor_params)
        self._strategy_predictor.start_predictor()

    def ingest_latest_data(self, new_data: RaceDataPacket):
        self._validate_race_data(new_data)

        gap_delta_method = self._gap_delta_method

        self._strategy_predictor.update_race_state(new_data=new_data)

        self._main_simulator.live_state_updates(live_race_data=new_data, gap_delta_method=gap_delta_method)

    def _validate_race_data(self, new_data):
        if not isinstance(new_data, RaceDataPacket):
            self.start_strategy_engine()
            raise TypeError(f'Expected RaceDataPacket, got {type(new_data).__name__}')

    def _apply_strategy_predictions_to_simulation(self, wait_time=0.04, time_limit_till_warning=1):
        start_time = time.time()
        while True:
            predictions, prediction_lap = self._strategy_predictor.get_predictions(wait_time=wait_time)    
            if prediction_lap == self._main_simulator.simulation_starting_lap:
                for driver, strategy in predictions.items():
                    if strategy:
                        self._main_simulator.update_driver_race_strategy(driver, strategy['tyre_usage'])
                return True

            elapsed_time = time.time() - start_time
            # TODO I need to add logging here just view it
            if elapsed_time > time_limit_till_warning:
                print("Warning: Prediction application is taking longer than expected."
                      " There may be an issue in the synchronization.")

    def run_strategy_engine(self):
        self._apply_strategy_predictions_to_simulation()

        self._main_simulator.run_simulation()

    def get_simulation_results(self, format_json: bool = False, json_indent: Optional[int | str] = None):
        predicted_result = self._main_simulator.get_result_from_simulation_run(format_json=format_json,
                                                                             json_indent=json_indent)
        predicted_positions = self._main_simulator.get_predicted_finishing_positions(format_json=format_json,
                                                                                     json_indent=json_indent)
        predicted_gaps = self._main_simulator.get_result_for_race_trace_plotting(format_json=format_json,
                                                                                 json_indent=json_indent)
        return predicted_result, predicted_positions, predicted_gaps

    def send_zmq_message(self, data):
        if self._socket:
            try:
                self._socket.send_json(data)
            except zmq.ZMQError as e:
                raise ConnectionError(f"Failed to send message via ZeroMQ: {e}")
        else:
            raise RuntimeError("ZeroMQ socket is not initialized. Did you call start_strategy_engine()?")

    def stop_strategy_engine(self):
        if self._socket:
            self._socket.close()
        self._strategy_predictor.stop_predictor()

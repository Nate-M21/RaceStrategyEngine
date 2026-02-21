use crate::{
    driver::Driver,
    race_config::RaceConfiguration,
    race_simulation::{RaceSimulation, SimulationData, SimulationResult},
    traits::RaceSimulationCore,
    utils::BoundedStack,
};

use std::{
    collections::HashMap,
    f64::INFINITY,
    mem,
    sync::{Arc, Mutex, RwLock, atomic::AtomicBool},
    thread::{self, JoinHandle},
};

// Define a unique key for strategies
#[derive(PartialEq, Eq, Hash, Clone)]
struct StrategyKey {
    driver_name: String,
    strategy: Vec<(String, u8)>,
}

impl StrategyKey {
    pub fn get_key(driver_name: String, strategy: Vec<(String, u8)>) -> Self {
        Self {
            driver_name,
            strategy,
        }
    }
}
#[derive(Default)]
struct StrategyStats {
    count: u64,
    position_sum: u64,
    race_time_sum: f64,
}

struct BestStrategy {
    mean_position: f64,
    strategy: Option<Vec<(String, u8)>>,
}

impl Default for BestStrategy {
    fn default() -> Self {
        Self {
            mean_position: INFINITY,
            strategy: None,
        }
    }
}
struct SimulationResultPacket {
    result: SimulationResult,
    current_lap: u8,
}

struct SimulationDataPacket {
    strategy_data: SimulationData,
    current_lap: u8,
}

// Store stats by StrategyKey
type StatsMap = HashMap<u8, HashMap<StrategyKey, StrategyStats>>;

// Additional index to quickly find best strategies
type DriverStrategyMap = HashMap<u8, HashMap<String, BestStrategy>>;

pub struct RealTimeStrategy<R: RaceSimulationCore> {
    race_simulation: RaceSimulation<R>,
    shared_live_data: Arc<RwLock<SimulationDataPacket>>,
    current_lap: u8,
    results_stack: Arc<Mutex<BoundedStack<SimulationResultPacket>>>,
    worker_threads: Vec<JoinHandle<()>>,
    shutdown_flag: Arc<AtomicBool>,

    stats_map: Arc<Mutex<StatsMap>>,
    driver_strategy_map: Arc<Mutex<DriverStrategyMap>>,
}

impl<R: RaceSimulationCore> RealTimeStrategy<R> {
    pub fn new(
        race_sim: R,
        drivers: HashMap<String, Driver>,
        race_config: RaceConfiguration,
    ) -> Self {
        let results_stack = Arc::new(Mutex::new(BoundedStack::new(10_000)));

        let shared_live_data = SimulationDataPacket {
            strategy_data: Vec::with_capacity(22),
            current_lap: 1,
        };
        let shared_live_data = Arc::new(RwLock::new(shared_live_data));
        let current_lap = 1;
        let race_simulation = RaceSimulation::new(race_sim, drivers, race_config);

        Self {
            race_simulation,
            shared_live_data,
            current_lap,
            results_stack,
            worker_threads: Vec::with_capacity(16),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            stats_map: Arc::new(Mutex::new(HashMap::with_capacity(2))),
            driver_strategy_map: Arc::new(Mutex::new(HashMap::with_capacity(2))),
        }
    }

    pub fn start_strategy_engine(&mut self) {
        self.start_simulation_result_processing_thread();
        self.start_worker_threads(6);
    }

    fn start_worker_threads(&mut self, thread_count: u8) {
        println!("Starting {thread_count} threads ");

        for i in 1..=thread_count {
            let simulation_data = Arc::clone(&self.shared_live_data);
            let result_stack = Arc::clone(&self.results_stack);
            let shutdown = Arc::clone(&self.shutdown_flag);

            let mut race_sim = self.race_simulation.clone();

            let handle = thread::spawn(move || {
                while !shutdown.load(std::sync::atomic::Ordering::Acquire) {
                    let data = simulation_data.read().unwrap();
                    let simulation_data = data.strategy_data.to_vec();
                    let current_lap = data.current_lap;
                    let mut simulation_data_map = HashMap::with_capacity(22);

                    for mut driver in simulation_data {
                        let name = mem::take(&mut driver.name);
                        simulation_data_map.insert(name, driver);
                    }
                    // TODO create API for this, prelim tests seem to say its faster.
                    race_sim
                        .race_sim_core
                        .incorporate_driver_data(simulation_data_map);
                    race_sim.race_sim_core.set_simulation_starting_point();
                    race_sim.race_sim_core.run_simulation();
                    let race_result = race_sim.race_sim_core.get_simulation_result();
                    // let race_result = race_sim.run_simulation(simulation_data);

                    let sim_result = SimulationResultPacket {
                        result: race_result,
                        current_lap,
                    };

                    result_stack.lock().unwrap().push(sim_result);
                    // sleep(Duration::new(5, 1));
                }
                println!("Shutting down Thread {i}");
            });
            self.worker_threads.push(handle);
        }
        println!("Completed starting {thread_count} threads ");
    }

    pub fn stop_strategy_engine(&mut self) {
        self.shutdown_flag
            .store(true, std::sync::atomic::Ordering::Release);
        println!("Sent message to shutdown all threads!");

        let threads = std::mem::take(&mut self.worker_threads);

        // Wait for each thread to finish
        for handle in threads {
            if let Err(e) = handle.join() {
                eprintln!("Error joining thread: {:?}", e);
            }
        }

        println!("Shutdown all threads");
        // thread::sleep(Duration::from_secs(2));
    }

    pub fn ingest_new_data(&mut self, strategy_data: SimulationData, current_lap: u8) {
        self.clean_up_old_laps(current_lap);
        let mut shared_live_data = self.shared_live_data.write().unwrap();

        self.current_lap = current_lap;

        *shared_live_data = SimulationDataPacket {
            strategy_data,
            current_lap,
        }
    }

    fn start_simulation_result_processing_thread(&mut self) {
        let shutdown = Arc::clone(&self.shutdown_flag);
        let results_stack = Arc::clone(&self.results_stack);

        let stats_map = Arc::clone(&self.stats_map);
        let driver_strategy_map = Arc::clone(&self.driver_strategy_map);

        let handle = thread::spawn(move || {
            let mut counter = 0;
            while !shutdown.load(std::sync::atomic::Ordering::Acquire) {
                let result = results_stack.lock().unwrap().pop();

                if let Some(race_result) = result {
                    RealTimeStrategy::<R>::store_simulation_result(
                        race_result,
                        &stats_map,
                        &driver_strategy_map,
                    );
                    counter += 1;
                }
            }

            println!("Amount completed {counter}")
        });
        self.worker_threads.push(handle);
    }

    fn store_simulation_result(
        sim_result: SimulationResultPacket,
        stats_map: &Arc<Mutex<StatsMap>>,
        driver_strategy_map: &Arc<Mutex<DriverStrategyMap>>,
    ) {
        let mut stats_map = stats_map.lock().unwrap();
        let mut driver_strategy_map = driver_strategy_map.lock().unwrap();

        let race_result = sim_result.result;
        let current_lap = sim_result.current_lap;

        for driver in race_result {
            let driver_name = driver.name;
            let position = driver.driver_position;
            let race_time = driver.driver_race_time;
            let strategy = driver.strategy;

            let key = StrategyKey::get_key(driver_name.clone(), strategy.clone());

            let lap_map = stats_map.entry(current_lap).or_insert_with(HashMap::new);
            let strategy_stats = lap_map.entry(key).or_insert_with(StrategyStats::default);

            strategy_stats.count += 1;
            strategy_stats.position_sum += position as u64;
            strategy_stats.race_time_sum += race_time;

            let new_mean_position =
                strategy_stats.position_sum as f64 / strategy_stats.count as f64;

            let best_strategies_map = driver_strategy_map
                .entry(current_lap)
                .or_insert_with(|| HashMap::with_capacity(22));
            let best_strategy = best_strategies_map
                .entry(driver_name)
                .or_insert_with(BestStrategy::default);

            if new_mean_position < best_strategy.mean_position {
                best_strategy.strategy = Some(strategy);
                best_strategy.mean_position = new_mean_position;
            }
        }
    }

    pub fn get_predictions(&self) -> Option<HashMap<String, Vec<(String, u8)>>> {
        let current_lap = self.current_lap;
        let driver_strategy_map = self.driver_strategy_map.lock().unwrap();

        if let Some(driver_map) = driver_strategy_map.get(&current_lap) {
            let mut best_strategies = HashMap::with_capacity(22);
            for (driver_name, best_strategy) in driver_map {
                let strategy = match &best_strategy.strategy {
                    Some(strategy) => strategy.clone(),
                    None => {
                        panic!("Turn on strategy engine before attempting to get predictions")
                    }
                };
                best_strategies.insert(driver_name.clone(), strategy);
            }
            return Some(best_strategies);
        }

        None
    }

    fn clean_up_old_laps(&mut self, current_lap: u8) {
        let previous_lap = current_lap - 1;
        self.driver_strategy_map
            .lock()
            .unwrap()
            .remove(&previous_lap);
        self.stats_map.lock().unwrap().remove(&previous_lap);
    }
}

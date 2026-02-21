use std::collections::{HashMap, VecDeque};
use std::f32::NEG_INFINITY;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::race_simulation::{DriverSimData, SimulationData};
use crate::{driver::Driver, race_config::RaceConfiguration};
use rand::random_range;
use rand::seq::IteratorRandom;
use rand::{
    rng,
    seq::{IndexedRandom, SliceRandom},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedStack<T> {
    pub data: VecDeque<T>,
    pub max_size: usize,
}

impl<T> BoundedStack<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn push(&mut self, value: T) {
        if self.len() >= self.max_size {
            self.data.pop_front();
        }

        self.data.push_back(value);
    }
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop_back()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.max_size
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, T> {
        self.data.iter()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

pub fn create_drivers_with_random_strategies(
    simulation_data: &SimulationData,
    alternate_strategies: &HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
    drivers_hash_map: &HashMap<String, Driver>,
    race_config: &RaceConfiguration,
) -> Vec<Driver> {
    let mut rng = rng();
    let mut drivers: Vec<Driver> = Vec::with_capacity(22);

    for data in simulation_data {
        let driver_name = &data.name;

        let mut driver = drivers_hash_map[driver_name].clone();

        let strategies = &alternate_strategies[driver_name];

        let mut strategy = select_random_strategy(strategies).clone();

        strategy.shuffle(&mut rng);

        modify_stints_randomly(&mut strategy, race_config, (-10, 10));
        let driver_accumulated_lap_times = Vec::new();
        let sim_data = DriverSimData::new_from_params(data, strategy, driver_accumulated_lap_times);
        driver.setup_for_simulation(sim_data, race_config);

        drivers.push(driver);
    }

    drivers
}

pub fn select_random_strategy(
    alternate_strategies: &HashMap<String, Vec<Vec<(String, u8)>>>,
) -> &Vec<(String, u8)> {
    let mut rng = rng();
    let stop_count = alternate_strategies.keys().choose(&mut rng).unwrap();

    let selected_strategy = alternate_strategies[stop_count].choose(&mut rng).unwrap();

    selected_strategy
}
pub fn modify_stints_randomly(
    strategy: &mut Vec<(String, u8)>,
    race_config: &RaceConfiguration,
    lap_variation_range: (i8, i8),
) {
    let mut temp_stints = Vec::with_capacity(10);

    for (_, stint) in strategy.iter() {
        let mut num = random_range(lap_variation_range.0..=lap_variation_range.1) as f64;
        num = *stint as f64 + num;
        let new_lap = f64::max(1.0, num);
        temp_stints.push(new_lap);
    }

    let new_stints = smart_round(temp_stints, race_config.num_laps as i32, true);

    for (index, (_compound, stint)) in strategy.iter_mut().enumerate() {
        *stint = new_stints[index] as u8
    }
}

pub fn smart_round(list: Vec<f64>, target_sum: i32, scale_by_contribution: bool) -> Vec<i32> {
    // Step 1: Scale the numbers if required
    let numbers = if scale_by_contribution {
        scale_values(list, target_sum as f64)
    } else {
        list
    };

    // Step 2: Round numbers and ensure each value is at least 1
    let mut rounded_numbers: Vec<i32> = numbers.iter().map(|&x| x.round() as i32).collect();
    for number in &mut rounded_numbers {
        if *number < 1 {
            *number = 1;
        }
    }

    let mut current_sum: i32 = rounded_numbers.iter().sum();

    // Step 3: Adjust rounded numbers to match target sum
    while current_sum != target_sum {
        let differences: Vec<f64> = numbers
            .iter()
            .zip(rounded_numbers.iter())
            .map(|(&orig, &rounded)| orig - rounded as f64)
            .collect();

        if current_sum < target_sum {
            // Need to increase the sum, find the best candidate to round up
            let min_diff_index = differences
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            rounded_numbers[min_diff_index] += 1;
            current_sum += 1;
        } else if current_sum > target_sum {
            // Need to decrease the sum, find the best candidate to round down
            // Only consider indices where rounded_numbers > 1
            let valid_indices: Vec<usize> = rounded_numbers
                .iter()
                .enumerate()
                .filter(|&(_, &val)| val > 1)
                .map(|(i, _)| i)
                .collect();

            if valid_indices.is_empty() {
                // If no valid indices, break to prevent infinite loop
                break;
            }

            let max_diff_index = valid_indices
                .iter()
                .max_by(|&&i, &&j| differences[i].partial_cmp(&differences[j]).unwrap())
                .unwrap();

            rounded_numbers[*max_diff_index] -= 1;
            current_sum -= 1;
        }
    }

    rounded_numbers
}

pub fn scale_values(numbers: Vec<f64>, target_number: f64) -> Vec<f64> {
    let total_sum: f64 = numbers.iter().sum();
    let scaling_factor = target_number / total_sum;
    numbers.iter().map(|&num| num * scaling_factor).collect()
}

// Utility function to save complete simulation state

pub fn save_complete_simulation_state(
    race_config: &RaceConfiguration,
    drivers: &HashMap<String, Driver>,
    sim_data: &SimulationData,
    path: &str,
) -> std::io::Result<()> {
    let state = (race_config, drivers, sim_data);
    let json = serde_json::to_string(&state).unwrap();
    std::fs::write(path, json)
}

pub fn load_complete_simulation_state(
    path: &str,
) -> std::io::Result<(RaceConfiguration, HashMap<String, Driver>, SimulationData)> {
    let data = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data).unwrap())
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RunningMeanStd {
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
    pub m2: Vec<f32>,
    pub count: usize,
    epsilon: f32,
}

impl RunningMeanStd {
    pub fn new(shape_size: usize, epsilon: f32) -> Self {
        Self {
            mean: vec![0.0; shape_size],
            variance: vec![1.0; shape_size],
            m2: vec![0.0; shape_size],
            count: epsilon as usize,
            epsilon,
        }
    }
    fn running_mean(old_mean: f32, new_value: f32, count: f32) -> f32 {
        let new_mean = old_mean + (new_value - old_mean) / count;

        new_mean
    }

    fn m2_calc(m2: f32, new_value: f32, new_mean: f32, old_mean: f32) -> f32 {
        let m2 = m2 + (new_value - old_mean) * (new_value - new_mean);

        m2
    }

    fn update_running_stats(&mut self, batch: &Vec<f32>) {
        self.count += 1;
        for (index, mean) in self.mean.iter_mut().enumerate() {
            let new_value = batch[index];
            // Update sum of squared differences
            // M2 = M2 + (new_value - old_mean) * (new_value - new_mean)
            let old_m2 = self.m2[index];
            let old_mean = *mean;
            let count = self.count as f32;

            let new_mean = RunningMeanStd::running_mean(old_mean, new_value, count);
            let new_m2 = RunningMeanStd::m2_calc(old_m2, new_value, new_mean, old_mean);

            let new_variance = new_m2 / count;

            *mean = new_mean;
            self.m2[index] = new_m2;

            self.variance[index] = new_variance;
        }
    }

    pub fn update_and_normalize(&mut self, batch: Vec<f32>) -> Vec<f32> {
        // println!("The length of the batch is: {}", batch.len());
        self.update_running_stats(&batch);
        self.apply_normalization(batch)
    }

    fn apply_normalization(&self, batch: Vec<f32>) -> Vec<f32> {
        // println!("The length of the batch is: {}", batch.len());

        batch
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let std_dev = f32::sqrt(self.variance[index] + self.epsilon);
                (value - self.mean[index]) / std_dev
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedNormalization {
    #[serde(skip)]
    pub rms: Arc<RwLock<RunningMeanStd>>,
    inference_only: bool,
}

impl SharedNormalization {
    pub fn new(observation_size: usize, inference_only: bool) -> Self {
        let rms = Arc::new(RwLock::new(RunningMeanStd::new(observation_size, 1e-4)));
        Self {
            rms,
            inference_only,
        }
    }

    fn update_and_normalize(&self, obs: Vec<f32>) -> Vec<f32> {
        let mut norm_vals = self.rms.write().unwrap().update_and_normalize(obs);

        // Clipping of (-1, 1)
        clip_values(&mut norm_vals);

        norm_vals
    }

    fn normalize_only(&self, obs: Vec<f32>) -> Vec<f32> {
        let mut norm_vals = self.rms.read().unwrap().apply_normalization(obs);

        clip_values(&mut norm_vals);

        norm_vals
    }

    pub fn normalize(&self, obs: Vec<f32>) -> Vec<f32> {
        if self.should_use_inference_only() {
            self.normalize_only(obs)
        } else {
            self.update_and_normalize(obs)
        }
    }

    fn should_use_inference_only(&self) -> bool {
        let threshold = self.rms.read().unwrap().count;
        let greater_than_threshold = threshold >= 5_000_000;
        self.inference_only || greater_than_threshold
    }
}

fn clip_values(norm_vals: &mut Vec<f32>) {
    for value in norm_vals.iter_mut() {
        *value = value.clamp(-10.0, 10.0);
    }
}

impl SharedNormalization {
    pub fn save_stats(&self, path: &Path) {
        let rms = self.rms.read().unwrap();
        let json = serde_json::to_string_pretty(&*rms).expect("Failed to convert to JSON");
        std::fs::write(path, json).expect("Failed to file");
    }

    pub fn load_stats(&self, path: &Path) {
        let json = match std::fs::read_to_string(path) {
            Ok(path) => path,
            Err(e) => panic!("Error for path - {path:?}: {e}"),
        };
        let loaded_rms: RunningMeanStd = match serde_json::from_str(&json) {
            Ok(val) => val,
            Err(e) => panic!("Error: {e}"),
        };
        let mut rms = self.rms.write().unwrap();
        if self.inference_only && loaded_rms.count < 1000 {
            panic!(
                "Cannot use inference-only normalization with only {} samples.
            Inference mode requires pre-computed statistics from sufficient training data.
            Either: 1) Train without inference_only=true first, or 2)
            Load stats file with more samples.",
                loaded_rms.count
            );
        };
        *rms = loaded_rms;
    }
}
pub fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0;
    let mut max_value = NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > max_value {
            best_index = index;
            max_value = *value;
        }
    }
    best_index
}

pub fn get_f1_points(position: u8) -> u8 {
    match position {
        1 => 25,
        2 => 18,
        3 => 15,
        4 => 12,
        5 => 10,
        6 => 8,
        7 => 6,
        8 => 4,
        9 => 2,
        10 => 1,
        _ => 0,
    }
}

pub fn unscale_value_1k(x: f32) -> f32 {
    x * 1000.0 // Inverse of scaling
}

pub fn scale_value_1k(x: f32) -> f32 {
    x / 1000.0 // Scale to roughly [-0.7, 2.3] range
}

pub fn create_fully_connected_edge_index(
    num_nodes: usize,
    include_self_loops: bool,
) -> Vec<(usize, usize)> {
    let mut connected_edges = Vec::with_capacity(num_nodes);

    for nodex_index in 0..num_nodes {
        for edge in 0..num_nodes {
            if !include_self_loops && nodex_index == edge {
            } else {
                connected_edges.push((nodex_index, edge));
            }
        }
    }

    connected_edges
}

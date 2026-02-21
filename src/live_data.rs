use std::collections::HashMap;

use serde::Serialize;

use crate::utils::argmax;

#[derive(Serialize)]
pub struct StrategyUpdate {
    driver: String,
    lap: u8,
    current_compound: String,
    current_position: u8,
    compound_compliant: bool,
    tyre_age: u8,
    pit_stops: u8,
    state_value: f32,
    confidence_spread: f32,
    positions: Vec<f32>,             // Full array of position probabilities
    strategies: Vec<StrategyOption>, // Top 5 recommended moves
    pit_windows: Vec<PitWindow>,     // Aggregated window confidence
    legal_actions: Vec<f32>,
    agent_attention: Option<Vec<AgentAttentionSource>>, // Who agent is watching
    attention_analysis: Option<AttentionAnalysis>,      // Who everyone is watching
    full_strategy: Vec<Stint>,
    strategy_compounds: Vec<String>,
    lap_analysis: Option<LapAttentionAnalysis>,
}

#[derive(Serialize)]
struct Stint {
    compound: String,
    laps: u8,
}

#[derive(Serialize)]
pub struct AgentAttentionSource {
    driver_name: String,
    driver_index: usize,
    attention_weight: f32,
}

#[derive(Serialize, Debug)]
pub struct DriverAttentionMetrics {
    driver_name: String,
    driver_index: usize,

    // Times ranked #1 by other drivers
    consensus_votes: usize,

    // Mean rank position (lower is better, 1.0 = always #1, 20.0 = always #20)
    mean_rank: f32,

    // Which driver this one watches most
    watching_driver_index: usize,
    watching_driver_name: String,
}

#[derive(Serialize, Debug)]
pub struct AttentionAnalysis {
    // All drivers ranked by mean_rank (ascending - lower is better)
    driver_metrics: Vec<DriverAttentionMetrics>,

    // Key insights
    consensus_leader: String, // Driver with most #1 votes
    consensus_votes: usize,   // How many #1 votes they got
    most_consistent: String,  // Lowest mean_rank (best overall)
    most_consistent_rank: f32,
}

#[derive(Serialize, Debug)]
pub struct LapAttentionMetrics {
    lap_name: String,
    lap_index: usize,
    consensus_votes: usize,
    mean_rank: f32,
    watching_lap_index: usize,
    watching_lap_name: String,
    lap_attention: Vec<LapAttentionData>,
}

#[derive(Serialize, Debug)]
pub struct LapAttentionAnalysis {
    lap_metrics: Vec<LapAttentionMetrics>,

    // Key insights
    consensus_leader: String,
    consensus_votes: usize,
    most_consistent: String,
    most_consistent_rank: f32,
}

#[derive(Serialize, Debug)]
pub struct LapAttentionData {
    lap: u8,
    compound_name: String,
    weight: f32,
    rank: u8,
}

#[derive(Serialize, Debug)]
pub struct DriverInfluence {
    rank: usize,
    driver_name: String,
    driver_index: usize,
    influence_score: f32,
}

#[derive(Serialize)]
struct StrategyOption {
    rank: usize,
    lap: Option<u8>,
    compound: String,
    confidence: f32,
}

#[derive(Serialize)]
struct PitWindow {
    start: u8,
    end: u8,
    confidence: f32,
    compound: String,
}

pub fn build_strategy_update(
    driver_name: &str,
    action_space: usize,
    num_laps: u8,
    priors: &[f32],
    state_value: &[f32],
    positions: &[f32],
    current_lap: u8,
    legal_actions: &[f32],
    compounds: &[&str; 3],
    current_compound: &str,
    tyre_age: u8,
    number_of_stops: u8,
    current_position: u8,
    compound_compliant: bool,
    driver_attention_weights: Option<(Vec<(usize, usize)>, Vec<Vec<f32>>)>,
    lap_attention_weights: Option<(Vec<(usize, usize)>, Vec<Vec<f32>>)>,
    predicted_strategy: Vec<(String, u8)>,
    predicted_compounds: Vec<String>,
    driver_names: &[String],
) -> StrategyUpdate {
    let mut indexed_probs: Vec<(usize, f32)> = priors
        .iter()
        .enumerate()
        .filter(|(i, _)| legal_actions[*i] > 0.0)
        .map(|(i, &p)| (i, p))
        .collect();

    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let max_prob = indexed_probs.first().map(|x| x.1).unwrap_or(1.0);
    let min_in_top5 = indexed_probs.get(4).map(|x| x.1).unwrap_or(0.0);
    let spread = (max_prob - min_in_top5) * 100.0;

    let no_pit_action = action_space - 1;

    let strategies: Vec<StrategyOption> = indexed_probs
        .iter()
        .take(5)
        .enumerate()
        .map(|(rank, &(action, prob))| {
            let (lap, compound_str) = if action == no_pit_action {
                (None, "NO PIT".to_string())
            } else {
                let lap_val = (action / 3) + 1;
                let comp = compounds[action % 3].to_uppercase();
                (Some(lap_val as u8), comp)
            };

            StrategyOption {
                rank: rank + 1,
                lap, // Stores Some(24) or None
                compound: compound_str,
                confidence: prob,
            }
        })
        .collect();

    let ranges = vec![
        (current_lap, current_lap + 5),
        (current_lap + 6, current_lap + 10),
        (current_lap + 11, current_lap + 15),
        (current_lap + 16, num_laps),
    ];

    let mut pit_windows = Vec::new();

    for (start, end) in ranges {
        if start > num_laps {
            continue;
        }
        let actual_end = end.min(num_laps);

        let mut max_conf = 0.0;
        let mut best_compound = "None";

        for lap in start..=actual_end {
            for (comp_idx, &compound) in compounds.iter().enumerate() {
                let action = ((lap - 1) * 3 + comp_idx as u8) as usize;

                if action < priors.len() && legal_actions[action] > 0.0 {
                    let conf = priors[action];
                    if conf > max_conf {
                        max_conf = conf;
                        best_compound = compound;
                    }
                }
            }
        }

        if max_conf > 0.01 {
            pit_windows.push(PitWindow {
                start,
                end: actual_end,
                confidence: max_conf,
                compound: best_compound.to_string(),
            });
        }
    }

    // Process attention weights if provided
    let (agent_attention, attention_analysis) =
        if let Some((edge_index, weights)) = driver_attention_weights {
            let agent_attn = calculate_agent_attention(&edge_index, &weights, driver_names);
            let driver_infl = perform_attention_analysis(&edge_index, &weights, driver_names);
            (Some(agent_attn), Some(driver_infl))
        } else {
            (None, None)
        };

    let lap_analysis = if let Some((edge_index, weights)) = &lap_attention_weights {
        let lap_infl = perform_lap_attention_analysis(edge_index, weights, &predicted_compounds);
        Some(lap_infl)
    } else {
        None
    };

    let full_strategy: Vec<Stint> = predicted_strategy
        .into_iter()
        .map(|(compound, laps)| Stint {
            compound: compound.to_uppercase(), // e.g., "SOFT"
            laps,
        })
        .collect();

    StrategyUpdate {
        driver: driver_name.to_string(),
        lap: current_lap,
        current_compound: current_compound.to_uppercase(),
        tyre_age,
        pit_stops: number_of_stops,
        state_value: state_value[0],
        confidence_spread: spread,
        positions: positions.to_vec(),
        strategies,
        pit_windows,
        current_position,
        compound_compliant,
        legal_actions: legal_actions.to_vec(),
        agent_attention,
        attention_analysis,
        full_strategy,
        strategy_compounds: predicted_compounds,
        lap_analysis,
    }
}

const AGENT_INDEX: usize = 0;
/// Calculate which drivers the agent (driver 0) is paying attention to
fn calculate_agent_attention(
    edge_index: &[(usize, usize)],
    weights: &[Vec<f32>],
    driver_names: &[String],
) -> Vec<AgentAttentionSource> {
    let mut agent_attention_to_incoming_info: Vec<(usize, f32)> = edge_index
        .iter()
        .zip(weights.iter())
        .filter(|&((_, target), _)| *target == AGENT_INDEX)
        .map(|((source, _), weights)| {
            // Average across attention heads
            let avg_weight = weights.iter().sum::<f32>() / weights.len() as f32;
            (*source, avg_weight)
        })
        .collect();

    // Sort by attention weight descending
    agent_attention_to_incoming_info.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    agent_attention_to_incoming_info
        .into_iter()
        .map(|(driver_idx, weight)| AgentAttentionSource {
            driver_name: driver_names
                .get(driver_idx)
                .cloned()
                .unwrap_or_else(|| format!("Driver {}", driver_idx)),
            driver_index: driver_idx,
            attention_weight: weight,
        })
        .collect()
}

/// Calculate which drivers are receiving the most attention overall (influence ranking)
fn perform_attention_analysis(
    edge_index: &[(usize, usize)],
    weights: &[Vec<f32>],
    driver_names: &[String],
) -> AttentionAnalysis {
    // Determine number of drivers from max index
    let num_drivers = driver_names.len();

    // which driver had the most number of times they were the highest
    // let mut driver_most_influence = HashMap::with_capacity(22);
    let mut consensus_votes = HashMap::with_capacity(num_drivers);

    // the rank of each driver i should divide by number of drivers but for now low is better
    // let mut driver_attention_ranker = HashMap::with_capacity(22);

    let mut rank_sum = HashMap::with_capacity(num_drivers);

    // the driver each driver is worried about most, ie which driver is going influence the momst drivers if theyve selcted
    // it or i can see which driver the model thinks will impact who the most
    // let mut driver_max_attention = HashMap::with_capacity(22);

    let mut watching_driver = HashMap::with_capacity(num_drivers);

    for index in 0..num_drivers {
        consensus_votes.insert(index, 0);
        rank_sum.insert(index, 0);

        watching_driver.insert(index, 0);
    }
    for index in 0..num_drivers {
        let driver_attention_to_incoming_info: Vec<(usize, f32)> = edge_index
            .iter()
            .zip(weights.iter())
            .filter(|&((_, target), _)| *target == index)
            .map(|((source, _), weights)| {
                // Average across attention heads
                let avg_weight = weights.iter().sum::<f32>() / weights.len() as f32;
                (*source, avg_weight)
            })
            .collect();

        let mut values = Vec::with_capacity(22);

        for (_index, value) in driver_attention_to_incoming_info {
            values.push(value);
        }

        // this is like reverse argsort so i can find which index has the highest value,
        // basically which driver is this driver source paying the most attention to
        let mut indices: Vec<usize> = (0..values.len()).collect();

        indices.sort_by(|&i, &j| values[j].total_cmp(&values[i])); // reversed i and j

        for (driver_rank, index_of_driver) in indices.iter().enumerate() {
            // which index is ranked the highest since it iter starts at index 0 adding +1
            let val = driver_rank + 1;
            *rank_sum.get_mut(index_of_driver).unwrap() += val;
        }

        let highest_index = argmax(&values);
        *watching_driver.get_mut(&index).unwrap() = highest_index;

        *consensus_votes.get_mut(&highest_index).unwrap() += 1;
    }

    // Accumulate incoming attention for each driver
    let mut incoming_attention = vec![0.0f32; num_drivers];

    for ((source, _target), weights) in edge_index.iter().zip(weights.iter()) {
        let avg_weight = weights.iter().sum::<f32>() / weights.len() as f32;
        incoming_attention[*source] += avg_weight;
    }

    // Build per-driver metrics
    let mut metrics: Vec<DriverAttentionMetrics> = (0..num_drivers)
        .map(|idx| {
            let watching_idx = *watching_driver.get(&idx).unwrap();
            DriverAttentionMetrics {
                driver_name: driver_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Driver {}", idx)),
                driver_index: idx,
                consensus_votes: *consensus_votes.get(&idx).unwrap(),
                mean_rank: *rank_sum.get(&idx).unwrap() as f32 / num_drivers as f32,
                watching_driver_index: watching_idx,
                watching_driver_name: driver_names
                    .get(watching_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Driver {}", watching_idx)),
            }
        })
        .collect();

    // Sort by mean_rank (ascending - lower is better)
    metrics.sort_by(|a, b| a.mean_rank.partial_cmp(&b.mean_rank).unwrap());

    // Extract key insights
    let consensus_leader_idx = consensus_votes
        .iter()
        .max_by_key(|&(_, &votes)| votes)
        .map(|(&idx, _)| idx)
        .unwrap_or(0);

    let most_consistent_idx = rank_sum
        .iter()
        .min_by_key(|&(_, &sum)| sum)
        .map(|(&idx, _)| idx)
        .unwrap_or(0);

    AttentionAnalysis {
        driver_metrics: metrics,
        consensus_leader: driver_names
            .get(consensus_leader_idx)
            .cloned()
            .unwrap_or_else(|| format!("Driver {}", consensus_leader_idx)),
        consensus_votes: *consensus_votes.get(&consensus_leader_idx).unwrap(),
        most_consistent: driver_names
            .get(most_consistent_idx)
            .cloned()
            .unwrap_or_else(|| format!("Driver {}", most_consistent_idx)),
        most_consistent_rank: *rank_sum.get(&most_consistent_idx).unwrap() as f32
            / num_drivers as f32,
    }
}

/// Calculate which drivers are receiving the most attention overall (influence ranking)
fn perform_lap_attention_analysis(
    edge_index: &[(usize, usize)],
    weights: &[Vec<f32>],
    compound_names: &[String],
) -> LapAttentionAnalysis {
    // Determine number of drivers from max index
    let num_compounds = compound_names.len();

    // which driver had the most number of times they were the highest
    // let mut driver_most_influence = HashMap::with_capacity(22);
    let mut consensus_votes = HashMap::with_capacity(num_compounds);

    // the rank of each driver i should divide by number of drivers but for now low is better
    // let mut driver_attention_ranker = HashMap::with_capacity(22);

    let mut rank_sum = HashMap::with_capacity(num_compounds);

    // the driver each driver is worried about most, ie which driver is going influence the momst drivers if theyve selcted
    // it or i can see which driver the model thinks will impact who the most
    // let mut driver_max_attention = HashMap::with_capacity(22);

    let mut watching_lap = HashMap::with_capacity(num_compounds);

    // the attention the node / key gives to other nodes, going to unpack this and put the values in each
    // lap metric so for each lap i know how much it valued other laps
    let mut lap_attention = HashMap::with_capacity(num_compounds);

    for index in 0..num_compounds {
        consensus_votes.insert(index, 0);
        rank_sum.insert(index, 0);

        watching_lap.insert(index, 0);
    }
    for index in 0..num_compounds {
        let lap_attention_to_incoming_info: Vec<(usize, f32)> = edge_index
            .iter()
            .zip(weights.iter())
            .filter(|&((_, target), _)| *target == index)
            .map(|((source, _), weights)| {
                // Average across attention heads
                let avg_weight = weights.iter().sum::<f32>() / weights.len() as f32;
                (*source, avg_weight)
            })
            .collect();

        let mut values = Vec::with_capacity(num_compounds);
        let mut laps_attention_data = Vec::with_capacity(num_compounds);

        for (index, value) in lap_attention_to_incoming_info {
            values.push(value);

            let lap_node = (index + 1) as u8;
            let compound_name = compound_names[index].clone();
            let data = LapAttentionData {
                lap: lap_node,
                compound_name,
                weight: value,
                rank: 0,
            };

            laps_attention_data.push(data);
        }
        laps_attention_data.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .expect("Failed to sort weights of lap attention")
        });

        for (index, data) in laps_attention_data.iter_mut().enumerate() {
            data.rank = (index + 1) as u8;
        }

        let lap = index + 1;
        lap_attention.insert(lap, laps_attention_data);

        // this is like reverse argsort so i can find which index has the highest value,
        // basically which driver is this driver source paying the most attention to
        let mut indices: Vec<usize> = (0..values.len()).collect();

        indices.sort_by(|&i, &j| values[j].total_cmp(&values[i])); // reversed i and j

        for (lap_rank, index_of_lap) in indices.iter().enumerate() {
            // which index is ranked the highest since it iter starts at index 0 adding +1
            let val = lap_rank + 1;
            *rank_sum.get_mut(index_of_lap).unwrap() += val;

            // i think this correct the index of the lap is lap plus 1
        }

        let highest_index = argmax(&values);
        *watching_lap.get_mut(&index).unwrap() = highest_index;

        *consensus_votes.get_mut(&highest_index).unwrap() += 1;
    }

    let mut metrics: Vec<LapAttentionMetrics> = (0..num_compounds)
        .map(|idx| {
            let watching_idx = *watching_lap.get(&idx).unwrap();
            let lap_num = idx + 1;
            LapAttentionMetrics {
                lap_name: compound_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Lap {}", idx)),
                lap_index: idx,
                consensus_votes: *consensus_votes.get(&idx).unwrap(),
                mean_rank: *rank_sum.get(&idx).unwrap() as f32 / num_compounds as f32,
                watching_lap_index: watching_idx,
                watching_lap_name: compound_names
                    .get(watching_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Lap {}", watching_idx)),
                lap_attention: lap_attention.remove(&lap_num).unwrap_or_default(), // ADD THIS
            }
        })
        .collect();

    // Sort by mean_rank (ascending - lower is better)
    metrics.sort_by(|a, b| a.mean_rank.partial_cmp(&b.mean_rank).unwrap());

    // Extract key insights
    let consensus_leader_idx = consensus_votes
        .iter()
        .max_by_key(|&(_, &votes)| votes)
        .map(|(&idx, _)| idx)
        .unwrap_or(0);

    let most_consistent_idx = rank_sum
        .iter()
        .min_by_key(|&(_, &sum)| sum)
        .map(|(&idx, _)| idx)
        .unwrap_or(0);

    LapAttentionAnalysis {
        lap_metrics: metrics,
        consensus_leader: compound_names
            .get(consensus_leader_idx)
            .cloned()
            .unwrap_or_else(|| format!("Lap {}", consensus_leader_idx)),
        consensus_votes: *consensus_votes.get(&consensus_leader_idx).unwrap(),
        most_consistent: compound_names
            .get(most_consistent_idx)
            .cloned()
            .unwrap_or_else(|| format!("Lap {}", most_consistent_idx)),
        most_consistent_rank: *rank_sum.get(&most_consistent_idx).unwrap() as f32
            / num_compounds as f32,
    }
}

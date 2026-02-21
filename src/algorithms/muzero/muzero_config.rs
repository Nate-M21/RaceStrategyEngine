#[derive(Debug, Clone, Copy)]
pub struct MuzeroConfig {
    pub num_iterations: u32,
    pub episode_iterations: u32,
    pub temperature: f32,
    pub action_space: usize,

    pub max_moves: usize,
    pub discount: f32,

    pub num_simulation: u32,

    // Dirichlet parameters for prior exploration noise.
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,

    // Training paramters
    pub td_steps: u32,
    pub training_steps: u32,
    pub buffer_size: usize,
    pub num_unroll_steps: u32,
    pub batch_size: usize,
    pub checkpoint_iterval: u32,

    // learning rate schedule that exponential
    pub learning_rate_init: f32,
    pub learning_rate_decay_steps: f32,
    pub learning_rate_decay_rate: f32,

    // UCB formula
    pub pb_c_base: u32,
    pub pb_c_init: f32,

    // Optimizer
    pub weight_decay: f32,
    pub momentum: f32,

    // Reward Bounds
    pub known_maximum_reward: Option<f32>,
    pub known_minimum_reward: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct AlphaZeroConfig {
    pub num_searches: u32,
    pub exploration_constant: f32,
    pub num_iterations: u32,
    pub episode_iterations: u32,
    pub temperature: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,

    pub batch_size: usize,
    pub training_steps: u32,
    pub buffer_size: usize,
    pub learning_rate: f32,

    pub known_maximum_reward: Option<f32>,
    pub known_minimum_reward: Option<f32>,

    // Gumbel Specific Config
    pub use_gumbel: bool,
    /// Typically 16 for board games, or equal to action space if small.
    pub gumbel_sample_size: usize,
    /// Scaling factor for Q-values in the sigma transform. Typically 50.0.
    pub c_visit: f32,
    /// Scaling factor for Q-values. Typically 1.0.
    pub c_scale: f32,
}

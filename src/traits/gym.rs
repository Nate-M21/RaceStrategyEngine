use std::path::Path;

pub trait GymEnvironment: Send + Sync {
    type Observation;
    type Reward;
    type Terminated: Into<bool>;
    type Truncated: Into<bool>;
    type Info;

    fn reset(&mut self) -> (Self::Observation, Self::Info);
    fn step(
        &mut self,
        action: usize,
    ) -> (
        Self::Observation,
        Self::Reward,
        Self::Terminated,
        Self::Truncated,
        Self::Info,
    );
    fn get_current_step(&self) -> usize;
    fn get_current_significant_step(&self) -> usize;
    fn action_space(&self) -> usize;
    fn observation_size(&self) -> usize;
    fn single_step_obs_dim(&self) -> usize;
    fn stack_size(&self) -> usize;
    fn max_steps(&self) -> usize;
    fn save_norm_stats(&self, path: &Path);
    fn load_norm_stats(&self, path: &Path);
}
pub trait MCTSGymEnvironment: GymEnvironment + Clone {
    fn branch(&self) -> Self;
    fn get_legal_actions(&self) -> Vec<f32>;
    fn clear(&mut self);
    fn show_info(&self);

    fn get_current_encoded_strategy(&self) -> Vec<f32> {
        unimplemented!("This Env does not implement get strategy")
    }
}

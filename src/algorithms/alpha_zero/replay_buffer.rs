use rand::seq::IndexedRandom;
use std::cmp::min;

#[derive(Debug)]
pub struct ReplayBuffer<Transition> {
    max_length: usize,
    batch_size: usize,
    head: usize,
    buffer: Vec<Transition>,
}

impl<Transition: Clone> ReplayBuffer<Transition> {
    pub fn new(max_length: usize, batch_size: usize) -> Self {
        Self {
            max_length,
            batch_size,
            head: 0,
            buffer: Vec::with_capacity(max_length),
        }
    }

    pub fn add_episode(&mut self, episode: Vec<Transition>) {
        if self.buffer_full() {
            self.overwite(episode);
        } else {
            self.buffer.extend(episode);
        }
    }

    pub fn sample_batch(&self) -> Vec<Transition> {
        let batch_size = min(self.batch_size, self.buffer.len());
        let rng = &mut rand::rng();

        let batch = self
            .buffer
            .choose_multiple(rng, batch_size)
            .cloned()
            .collect::<Vec<Transition>>();

        batch
    }

    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }

    pub fn buffer_full(&self) -> bool {
        self.buffer.len() >= self.max_length
    }

    fn overwite(&mut self, episode: Vec<Transition>) {
        for transition in episode {
            self.buffer[self.head] = transition;
            self.move_head_next_position()
        }
    }

    fn move_head_next_position(&mut self) {
        if self.head == self.buffer.len() - 1 {
            self.head = 0
        } else {
            self.head += 1;
        }
    }
}

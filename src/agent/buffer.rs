use ndarray::Array1;
use rand::{prelude::SliceRandom, Rng};

pub struct BufferEntry {
    /// Information state of the entry
    pub info_state: Array1<f32>,
    /// Legal actions in the current state
    pub action_mask: Array1<f32>,
    /// Current step/iteration
    pub step: usize,
    /// Buffer data associated with the entry
    pub data: Array1<f32>,
}


pub struct ReservoirBuffer {
    /// Entries in the buffer
    entries: Vec<BufferEntry>,
}

impl ReservoirBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }
    pub fn add(&mut self, entry: BufferEntry) {
        if self.entries.len() < self.entries.capacity() {
            self.entries.push(entry);
        } else {
            // Randomly sample an entry to replace in the buffer
            let mut rng = rand::thread_rng();
            let idx = rng.gen_range(0..self.entries.len() - 1);
            self.entries[idx] = entry;
        }
    }

    pub fn sample(&self, num_entries: usize) -> Vec<&BufferEntry> {
        let mut rng = rand::thread_rng();
        self.entries
            .choose_multiple(&mut rng, num_entries)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}


impl std::ops::Index<usize> for ReservoirBuffer {
    type Output = BufferEntry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}
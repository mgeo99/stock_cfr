use super::buffer::ReservoirBuffer;
use rand::prelude::*;
use tch::{Tensor, Device};

pub struct ReservoirDataset<'a> {
    buffer: &'a ReservoirBuffer,
    shuffle_idx: Vec<usize>,
    device: Device,
    batch_idx: usize,
    num_batches: usize,
    pos: usize,
    batch_size: usize
}

impl<'a> ReservoirDataset<'a> {
    pub fn new(buffer: &'a ReservoirBuffer, device: Device, num_batches: usize, batch_size: usize) -> Self {
        let mut indices = (0..buffer.len()).collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        let num_batches = std::cmp::min(num_batches, buffer.len() / batch_size);
        Self {
            buffer,
            shuffle_idx: indices,
            device,
            batch_idx: 0,
            num_batches,
            pos: 0,
            batch_size
        }
    }

    pub fn next_batch(&mut self) -> Option<ReservoirBatch> {
        if self.batch_idx >= self.num_batches {
            return None;
        }

        let mut info_states: Vec<Tensor> = Vec::with_capacity(self.batch_size);
        let mut action_masks: Vec<Tensor> = Vec::with_capacity(self.batch_size);
        let mut observations: Vec<Tensor> = Vec::with_capacity(self.batch_size);
        let mut steps: Vec<Tensor> = Vec::with_capacity(self.batch_size);

        let indices = &self.shuffle_idx[self.pos..self.pos + self.batch_size];
        for &i in indices {
            let entry = &self.buffer[i];
            info_states.push(Tensor::of_slice(entry.info_state.as_slice().unwrap()));
            action_masks.push(Tensor::of_slice(entry.action_mask.as_slice().unwrap()));
            observations.push(Tensor::of_slice(entry.data.as_slice().unwrap()));
            steps.push(Tensor::of_slice(&[entry.step as f32]));
        }
        self.pos += self.batch_size;
        self.batch_idx += 1;

        let info_states = Tensor::stack(&info_states, 0).to(self.device);
        let action_masks = Tensor::stack(&action_masks, 0).to(self.device);
        let observations = Tensor::stack(&observations, 0).to(self.device);
        let steps = Tensor::stack(&steps, 0).to(self.device);
        
        Some(ReservoirBatch {
            info_states,
            action_masks,
            observations,
            steps
        })
    }

}


pub struct ReservoirBatch {
    /// Information States
    pub info_states: Tensor,
    /// Valid action masks
    pub action_masks: Tensor,
    /// Data associated with the entry in the batch (often times used as a label)
    pub observations: Tensor,
    /// Step each entry in the batch was collected at
    pub steps: Tensor
}
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Profiler {
    task_timings: HashMap<usize, Duration>,
    unit_utilization: HashMap<usize, f32>,
    memory_usage: Vec<usize>,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start_task(&mut self) -> Instant {
        Instant::now()
    }

    pub fn end_task(&mut self, task_id: usize, start_time: Instant) {
        let duration = start_time.elapsed();
        self.task_timings.insert(task_id, duration);
    }

    pub fn update_unit_utilization(&mut self, unit_id: usize, utilization: f32) {
        self.unit_utilization.insert(unit_id, utilization);
    }

    pub fn record_memory_usage(&mut self, usage: usize) {
        self.memory_usage.push(usage);
    }

    pub fn get_task_timing(&self, task_id: usize) -> Option<&Duration> {
        self.task_timings.get(&task_id)
    }

    pub fn get_unit_utilization(&self, unit_id: usize) -> Option<&f32> {
        self.unit_utilization.get(&unit_id)
    }

    pub fn get_average_memory_usage(&self) -> Option<f32> {
        if self.memory_usage.is_empty() {
            None
        } else {
            Some(self.memory_usage.iter().sum::<usize>() as f32 / self.memory_usage.len() as f32)
        }
    }
}

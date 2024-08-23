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

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_task_profiling() {
        let mut profiler = Profiler::new();
        let start_time = profiler.start_task();
        thread::sleep(Duration::from_millis(100));
        profiler.end_task(1, start_time);

        let task_timing = profiler.get_task_timing(1).unwrap();
        assert!(task_timing.as_millis() >= 100);
    }

    #[test]
    fn test_unit_utilization() {
        let mut profiler = Profiler::new();
        profiler.update_unit_utilization(1, 0.75);
        assert_eq!(profiler.get_unit_utilization(1), Some(&0.75));
        assert_eq!(profiler.get_unit_utilization(2), None);
    }

    #[test]
    fn test_memory_usage() {
        let mut profiler = Profiler::new();
        profiler.record_memory_usage(1000);
        profiler.record_memory_usage(2000);
        profiler.record_memory_usage(3000);

        assert_eq!(profiler.get_average_memory_usage(), Some(2000.0));
    }

    #[test]
    fn test_empty_profiler() {
        let profiler = Profiler::new();
        assert_eq!(profiler.get_task_timing(1), None);
        assert_eq!(profiler.get_unit_utilization(1), None);
        assert_eq!(profiler.get_average_memory_usage(), None);
    }
}

use std::collections::HashMap;
use std::time::Instant;

pub struct ResourceMonitor {
    cpu_usage: HashMap<String, f32>,
    memory_usage: HashMap<String, usize>,
    gpu_usage: HashMap<String, f32>,
    last_update: Instant,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        ResourceMonitor {
            cpu_usage: HashMap::new(),
            memory_usage: HashMap::new(),
            gpu_usage: HashMap::new(),
            last_update: Instant::now(),
        }
    }

    pub fn update_cpu_usage(&mut self, node_id: &str, usage: f32) {
        self.cpu_usage.insert(node_id.to_string(), usage);
        self.last_update = Instant::now();
    }

    pub fn update_memory_usage(&mut self, node_id: &str, usage: usize) {
        self.memory_usage.insert(node_id.to_string(), usage);
        self.last_update = Instant::now();
    }

    pub fn update_gpu_usage(&mut self, node_id: &str, usage: f32) {
        self.gpu_usage.insert(node_id.to_string(), usage);
        self.last_update = Instant::now();
    }

    pub fn get_cpu_usage(&self, node_id: &str) -> Option<f32> {
        self.cpu_usage.get(node_id).cloned()
    }

    pub fn get_memory_usage(&self, node_id: &str) -> Option<usize> {
        self.memory_usage.get(node_id).cloned()
    }

    pub fn get_gpu_usage(&self, node_id: &str) -> Option<f32> {
        self.gpu_usage.get(node_id).cloned()
    }

    pub fn last_update_time(&self) -> Instant {
        self.last_update
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_update_and_get_cpu_usage() {
        let mut monitor = ResourceMonitor::new();
        monitor.update_cpu_usage("node1", 0.75);
        assert_eq!(monitor.get_cpu_usage("node1"), Some(0.75));
        assert_eq!(monitor.get_cpu_usage("node2"), None);
    }

    #[test]
    fn test_update_and_get_memory_usage() {
        let mut monitor = ResourceMonitor::new();
        monitor.update_memory_usage("node1", 1024);
        assert_eq!(monitor.get_memory_usage("node1"), Some(1024));
        assert_eq!(monitor.get_memory_usage("node2"), None);
    }

    #[test]
    fn test_update_and_get_gpu_usage() {
        let mut monitor = ResourceMonitor::new();
        monitor.update_gpu_usage("node1", 0.9);
        assert_eq!(monitor.get_gpu_usage("node1"), Some(0.9));
        assert_eq!(monitor.get_gpu_usage("node2"), None);
    }

    #[test]
    fn test_last_update_time() {
        let mut monitor = ResourceMonitor::new();
        let initial_time = monitor.last_update_time();
        thread::sleep(Duration::from_millis(10));
        monitor.update_cpu_usage("node1", 0.5);
        assert!(monitor.last_update_time() > initial_time);
    }

    #[test]
    fn test_multiple_updates() {
        let mut monitor = ResourceMonitor::new();
        monitor.update_cpu_usage("node1", 0.5);
        monitor.update_memory_usage("node1", 2048);
        monitor.update_gpu_usage("node1", 0.8);

        assert_eq!(monitor.get_cpu_usage("node1"), Some(0.5));
        assert_eq!(monitor.get_memory_usage("node1"), Some(2048));
        assert_eq!(monitor.get_gpu_usage("node1"), Some(0.8));
    }
}

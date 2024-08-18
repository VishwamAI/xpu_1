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

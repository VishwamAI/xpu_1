use std::collections::HashMap;
use std::time::Instant;
use crate::power_management::PowerState;
use crate::task_scheduling::ProcessingUnitType;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XpuStatus {
    pub overall: SystemStatus,
    pub processing_units: HashMap<ProcessingUnitType, ProcessingUnitStatus>,
    pub memory: MemoryStatus,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemStatus {
    Running,
    Stopped,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingUnitStatus {
    pub utilization: f32,
    pub temperature: f32,
    pub power_state: PowerState,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStatus {
    pub usage: usize,
    pub total: usize,
    pub swap_usage: usize,
    pub swap_total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_load: LoadLevel,
    pub active_tasks: usize,
    pub queued_tasks: usize,
}

impl Default for SystemHealth {
    fn default() -> Self {
        SystemHealth {
            overall_load: LoadLevel::Low,
            active_tasks: 0,
            queued_tasks: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl LoadLevel {
    pub fn from_cpu_usage(usage: f32) -> Self {
        match usage {
            u if u < 0.3 => LoadLevel::Low,
            u if u < 0.6 => LoadLevel::Medium,
            u if u < 0.9 => LoadLevel::High,
            _ => LoadLevel::Critical,
        }
    }
}

pub struct ResourceMonitor {
    cpu_usage: HashMap<String, f32>,
    memory_usage: HashMap<String, usize>,
    gpu_usage: HashMap<String, f32>,
    swap_usage: HashMap<String, usize>,
    active_tasks: usize,
    queued_tasks: usize,
    last_update: Instant,
    power_states: HashMap<String, PowerState>,
    total_memory: usize,
    temperatures: HashMap<String, f32>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        ResourceMonitor {
            cpu_usage: HashMap::new(),
            memory_usage: HashMap::new(),
            gpu_usage: HashMap::new(),
            swap_usage: HashMap::new(),
            active_tasks: 0,
            queued_tasks: 0,
            last_update: Instant::now(),
            power_states: HashMap::new(),
            total_memory: 0,
            temperatures: HashMap::new(),
        }
    }

    pub fn get_power_state(&self, unit_id: &str) -> Option<&PowerState> {
        self.power_states.get(unit_id)
    }

    pub fn get_swap_usage(&self) -> usize {
        self.swap_usage.values().sum()
    }

    pub fn get_total_memory(&self) -> usize {
        self.total_memory
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

    pub fn update_swap_usage(&mut self, node_id: &str, usage: usize) {
        self.swap_usage.insert(node_id.to_string(), usage);
        self.last_update = Instant::now();
    }

    pub fn update_task_counts(&mut self, active: usize, queued: usize) {
        self.active_tasks = active;
        self.queued_tasks = queued;
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

    pub fn get_total_swap(&self) -> usize {
        self.swap_usage.values().sum()
    }

    pub fn get_active_tasks_count(&self) -> usize {
        self.active_tasks
    }

    pub fn get_queued_tasks_count(&self) -> usize {
        self.queued_tasks
    }

    pub fn last_update_time(&self) -> Instant {
        self.last_update
    }

    pub fn get_temperature(&self, unit_id: &str) -> Option<f32> {
        self.temperatures.get(unit_id).cloned()
    }

    pub fn update_temperature(&mut self, unit_id: &str, temperature: f32) {
        self.temperatures.insert(unit_id.to_string(), temperature);
        self.last_update = Instant::now();
    }

    pub fn get_usage(&self, unit_id: &str) -> Option<f32> {
        self.cpu_usage.get(unit_id).cloned()
            .or_else(|| self.gpu_usage.get(unit_id).cloned())
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

    #[test]
    fn test_update_and_get_temperature() {
        let mut monitor = ResourceMonitor::new();
        monitor.update_temperature("node1", 45.5);
        assert_eq!(monitor.get_temperature("node1"), Some(45.5));
        assert_eq!(monitor.get_temperature("node2"), None);
    }

    #[test]
    fn test_get_usage() {
        let mut monitor = ResourceMonitor::new();
        monitor.update_cpu_usage("cpu1", 0.6);
        monitor.update_gpu_usage("gpu1", 0.8);
        assert_eq!(monitor.get_usage("cpu1"), Some(0.6));
        assert_eq!(monitor.get_usage("gpu1"), Some(0.8));
        assert_eq!(monitor.get_usage("unknown"), None);
    }
}

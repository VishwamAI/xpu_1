use std::time::Duration;
use std::sync::{Arc, Mutex};
use xpu_manager_rust::{
    EnergyProfile, MemoryManager, PowerManager, PowerState, ProcessingUnitType,
    Task, Scheduler, XpuOptimizer, XpuOptimizerConfig, SchedulerType, MemoryManagerType,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xpu_optimizer_creation() -> Result<(), Box<dyn std::error::Error>> {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
            power_management_policy: "default".to_string(),
            cloud_offloading_policy: "default".to_string(),
            adaptive_optimization_policy: "default".to_string(),
        };
        let optimizer = XpuOptimizer::new(config)?;

        assert!(optimizer.task_queue.is_empty());
        let memory_manager = optimizer.memory_manager.lock().unwrap();
        assert_eq!(memory_manager.get_available_memory(), 1024);
        assert!(matches!(optimizer.power_manager.get_power_state(), PowerState::Normal));
        Ok(())
    }

    #[test]
    fn test_add_task() -> Result<(), Box<dyn std::error::Error>> {
        let config = XpuOptimizerConfig::default();
        let mut optimizer = XpuOptimizer::new(config)?;

        let task = Task::new(
            1,
            1,
            vec![],
            Duration::from_secs(1),
            100,
            false,
            ProcessingUnitType::CPU,
        );

        optimizer.add_task(task, "")?;
        assert_eq!(optimizer.task_queue.len(), 1);
        Ok(())
    }

    #[test]
    fn test_schedule_tasks() -> Result<(), Box<dyn std::error::Error>> {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            ..Default::default()
        };
        let mut optimizer = XpuOptimizer::new(config)?;

        // Add multiple tasks
        for i in 1..=5 {
            let task = Task::new(
                i,
                1,
                vec![],
                Duration::from_secs(1),
                100,
                false,
                ProcessingUnitType::CPU,
            );
            optimizer.add_task(task, "")?;
        }

        assert_eq!(optimizer.task_queue.len(), 5);
        optimizer.run()?;
        assert!(optimizer.task_queue.is_empty());
        Ok(())
    }

    #[test]
    fn test_manage_memory() -> Result<(), Box<dyn std::error::Error>> {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            ..Default::default()
        };
        let mut optimizer = XpuOptimizer::new(config)?;

        // Add tasks with memory requirements
        for i in 1..=5 {
            let task = Task::new(
                i,
                1,
                vec![],
                Duration::from_secs(1),
                100,
                false,
                ProcessingUnitType::CPU,
            );
            optimizer.add_task(task, "")?;
        }

        optimizer.allocate_memory_for_tasks()?;
        let memory_manager = optimizer.memory_manager.lock().unwrap();
        assert_eq!(memory_manager.get_available_memory(), 1024 - 5 * 100);
        Ok(())
    }
}

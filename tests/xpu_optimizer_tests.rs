use xpu_1::{XpuOptimizer, XpuOptimizerConfig, SchedulerType, MemoryManagerType, Task, ProcessingUnit, ProcessingUnitType};
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xpu_optimizer_creation() {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
        };
        let optimizer = XpuOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_add_task() {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
        };
        let mut optimizer = XpuOptimizer::new(config).unwrap();

        let task = Task {
            id: 1,
            unit: ProcessingUnit {
                unit_type: ProcessingUnitType::CPU,
                processing_power: 1.0,
                current_load: 0.0,
            },
            priority: 1,
            dependencies: vec![],
            execution_time: Duration::from_secs(1),
            memory_requirement: 100,
        };

        assert!(optimizer.add_task(task).is_ok());
    }

    #[test]
    fn test_remove_task() {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
        };
        let mut optimizer = XpuOptimizer::new(config).unwrap();

        let task = Task {
            id: 1,
            unit: ProcessingUnit {
                unit_type: ProcessingUnitType::CPU,
                processing_power: 1.0,
                current_load: 0.0,
            },
            priority: 1,
            dependencies: vec![],
            execution_time: Duration::from_secs(1),
            memory_requirement: 100,
        };

        optimizer.add_task(task).unwrap();
        let removed_task = optimizer.remove_task(1);
        assert!(removed_task.is_ok());
    }

    #[test]
    fn test_schedule_tasks() {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
        };
        let mut optimizer = XpuOptimizer::new(config).unwrap();

        // Add multiple tasks
        for i in 1..=5 {
            let task = Task {
                id: i,
                unit: ProcessingUnit {
                    unit_type: ProcessingUnitType::CPU,
                    processing_power: 1.0,
                    current_load: 0.0,
                },
                priority: 1,
                dependencies: vec![],
                execution_time: Duration::from_secs(1),
                memory_requirement: 100,
            };
            optimizer.add_task(task).unwrap();
        }

        assert!(optimizer.schedule_tasks().is_ok());
    }

    #[test]
    fn test_manage_memory() {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
        };
        let mut optimizer = XpuOptimizer::new(config).unwrap();

        // Add tasks with memory requirements
        for i in 1..=5 {
            let task = Task {
                id: i,
                unit: ProcessingUnit {
                    unit_type: ProcessingUnitType::CPU,
                    processing_power: 1.0,
                    current_load: 0.0,
                },
                priority: 1,
                dependencies: vec![],
                execution_time: Duration::from_secs(1),
                memory_requirement: 100,
            };
            optimizer.add_task(task).unwrap();
        }

        assert!(optimizer.manage_memory().is_ok());
    }
}

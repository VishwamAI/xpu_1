use xpu_optimization::{XpuOptimizer, XpuOptimizerConfig, SchedulerType, MemoryManagerType, Task, ProcessingUnit, ProcessingUnitType};
use std::time::Duration;

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn create_test_optimizer() -> XpuOptimizer {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024,
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
        };
        XpuOptimizer::new(config).unwrap()
    }

    fn create_test_task(id: usize) -> Task {
        Task {
            id,
            unit: ProcessingUnit {
                unit_type: ProcessingUnitType::CPU,
                processing_power: 1.0,
                current_load: 0.0,
            },
            priority: 1,
            dependencies: vec![],
            execution_time: Duration::from_secs(1),
            memory_requirement: 100,
        }
    }

    #[test]
    fn test_varying_task_loads() {
        let mut optimizer = create_test_optimizer();

        // Add tasks with varying loads
        for i in 1..=10 {
            let mut task = create_test_task(i);
            task.execution_time = Duration::from_secs(i as u64);
            task.memory_requirement = i * 50;
            assert!(optimizer.add_task(task).is_ok());
        }

        assert!(optimizer.schedule_tasks().is_ok());
        assert!(optimizer.manage_memory().is_ok());

        // Verify task distribution and memory allocation
        let scheduled_tasks = optimizer.get_scheduled_tasks();
        assert_eq!(scheduled_tasks.len(), 10);

        let memory_usage = optimizer.get_memory_usage();
        assert!(memory_usage > 0 && memory_usage <= optimizer.config.memory_pool_size);

        // Check if tasks are distributed across different processing units
        let unit_task_counts = optimizer.get_unit_task_counts();
        assert!(unit_task_counts.values().all(|&count| count > 0));
    }

    #[test]
    fn test_different_scheduling_algorithms() {
        let config_rr = XpuOptimizerConfig {
            scheduler_type: SchedulerType::RoundRobin,
            ..create_test_optimizer().config
        };
        let mut optimizer_rr = XpuOptimizer::new(config_rr).unwrap();

        let config_lb = XpuOptimizerConfig {
            scheduler_type: SchedulerType::LoadBalancing,
            ..create_test_optimizer().config
        };
        let mut optimizer_lb = XpuOptimizer::new(config_lb).unwrap();

        // Add same tasks to both optimizers
        for i in 1..=5 {
            let task = create_test_task(i);
            assert!(optimizer_rr.add_task(task.clone()).is_ok());
            assert!(optimizer_lb.add_task(task).is_ok());
        }

        assert!(optimizer_rr.schedule_tasks().is_ok());
        assert!(optimizer_lb.schedule_tasks().is_ok());

        // Compare scheduling results
        let rr_distribution = optimizer_rr.get_unit_task_counts();
        let lb_distribution = optimizer_lb.get_unit_task_counts();

        // Round Robin should distribute tasks evenly
        assert!(rr_distribution.values().all(|&count| count == 1 || count == 2));

        // Load Balancing should prefer less loaded units
        assert!(lb_distribution.values().any(|&count| count > 1));
        assert!(lb_distribution.values().any(|&count| count == 0));
    }

    #[test]
    fn test_memory_management_strategies() {
        let config_simple = XpuOptimizerConfig {
            memory_manager_type: MemoryManagerType::Simple,
            ..create_test_optimizer().config
        };
        let mut optimizer_simple = XpuOptimizer::new(config_simple).unwrap();

        let config_dynamic = XpuOptimizerConfig {
            memory_manager_type: MemoryManagerType::Dynamic,
            ..create_test_optimizer().config
        };
        let mut optimizer_dynamic = XpuOptimizer::new(config_dynamic).unwrap();

        // Add tasks with varying memory requirements
        for i in 1..=5 {
            let mut task = create_test_task(i);
            task.memory_requirement = i * 100;
            assert!(optimizer_simple.add_task(task.clone()).is_ok());
            assert!(optimizer_dynamic.add_task(task).is_ok());
        }

        assert!(optimizer_simple.manage_memory().is_ok());
        assert!(optimizer_dynamic.manage_memory().is_ok());

        // TODO: Add assertions to compare memory allocation results
    }
}

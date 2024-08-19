use std::time::Duration;
use xpu_manager_rust::{MemoryManager, PowerManager, PowerState, Task, TaskScheduler, ProcessingUnitType, ProcessingUnit, EnergyProfile};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xpu_optimizer_creation() {
        let num_processing_units = 4;
        let total_memory = 1024;
        let scheduler = TaskScheduler::new(num_processing_units);
        let memory_manager = MemoryManager::new(total_memory);
        let power_manager = PowerManager::new();

        assert!(scheduler.tasks.is_empty());
        assert_eq!(memory_manager.get_available_memory(), total_memory);
        assert!(matches!(
            power_manager.get_power_state(),
            PowerState::Normal
        ));
    }

    #[test]
    fn test_add_task() {
        let mut scheduler = TaskScheduler::new(4);

        let task = Task {
            id: 1,
            priority: 1,
            execution_time: Duration::from_secs(1),
            memory_requirement: 100,
            unit_type: ProcessingUnitType::CPU,
            dependencies: vec![],
            secure: false,
            unit: ProcessingUnit {
                id: 0,
                unit_type: ProcessingUnitType::CPU,
                current_load: Duration::new(0, 0),
                processing_power: 1.0,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            },
            estimated_duration: Duration::from_secs(2),
            estimated_resource_usage: 120,
        };

        scheduler.add_task(task);
        assert_eq!(scheduler.tasks.len(), 1);
    }

    // Test for removing a task is not applicable in the current implementation
    // as TaskScheduler does not have a direct method for removing tasks.
    // If task removal functionality is needed, it should be implemented
    // in the TaskScheduler struct and then tested here.

    #[test]
    fn test_schedule_tasks() {
        let num_processing_units = 4;
        let mut scheduler = TaskScheduler::new(num_processing_units);

        // Add multiple tasks
        for i in 1..=5 {
            let task = Task {
                id: i,
                priority: 1,
                execution_time: Duration::from_secs(1),
                memory_requirement: 100,
                unit_type: ProcessingUnitType::CPU,
                dependencies: vec![],
                secure: false,
                unit: ProcessingUnit {
                    id: i - 1,
                    unit_type: ProcessingUnitType::CPU,
                    current_load: Duration::new(0, 0),
                    processing_power: 1.0,
                    power_state: PowerState::Normal,
                    energy_profile: EnergyProfile::default(),
                },
                estimated_duration: Duration::from_secs(2), // Added estimated duration
                estimated_resource_usage: 120, // Added estimated resource usage
            };
            scheduler.add_task(task);
        }

        assert_eq!(scheduler.tasks.len(), 5);
        let _ = scheduler.schedule();
        assert!(scheduler.tasks.is_empty());
    }

    #[test]
    fn test_manage_memory() {
        let total_memory = 1024;
        let mut memory_manager = MemoryManager::new(total_memory);
        let mut task_scheduler = TaskScheduler::new(4);

        // Add tasks with memory requirements
        for i in 1..=5 {
            let task = Task {
                id: i,
                priority: 1,
                execution_time: Duration::from_secs(1),
                memory_requirement: 100,
                unit_type: ProcessingUnitType::CPU,
                dependencies: vec![],
                secure: false,
                unit: ProcessingUnit {
                    id: i - 1,
                    unit_type: ProcessingUnitType::CPU,
                    current_load: Duration::new(0, 0),
                    processing_power: 1.0,
                    power_state: PowerState::Normal,
                    energy_profile: EnergyProfile::default(),
                },
                estimated_duration: Duration::from_secs(2),
                estimated_resource_usage: 120,
            };
            task_scheduler.add_task(task);
        }

        assert!(memory_manager
            .allocate_for_tasks(task_scheduler.tasks.make_contiguous())
            .is_ok());
        assert_eq!(
            memory_manager.get_available_memory(),
            total_memory - 5 * 100
        );
    }
}

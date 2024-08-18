use xpu_manager_rust::{TaskScheduler, Task, MemoryManager, PowerManager, PowerState};
use std::time::Duration;

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
        assert!(matches!(power_manager.get_power_state(), PowerState::Normal));
    }

    #[test]
    fn test_add_task() {
        let mut scheduler = TaskScheduler::new(4);

        let task = Task {
            id: 1,
            priority: 1,
            execution_time: Duration::from_secs(1),
            memory_requirement: 100,
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
            };
            scheduler.add_task(task);
        }

        assert_eq!(scheduler.tasks.len(), 5);
        scheduler.schedule();
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
            };
            task_scheduler.add_task(task);
        }

        assert!(memory_manager.allocate_for_tasks(&task_scheduler.tasks).is_ok());
        assert_eq!(memory_manager.get_available_memory(), total_memory - 5 * 100);
    }
}

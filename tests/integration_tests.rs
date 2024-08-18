use std::time::Duration;
use xpu_manager_rust::{
    memory_management::MemoryManager,
    power_management::{PowerManager, PowerState},
    task_scheduling::{Task, TaskScheduler},
};

#[test]
fn test_task_scheduling_and_memory_allocation() {
    let num_processing_units = 2;
    let total_memory = 1000;
    let mut scheduler = TaskScheduler::new(num_processing_units);
    let mut memory_manager = MemoryManager::new(total_memory);

    let tasks = vec![
        Task {
            id: 1,
            priority: 2,
            execution_time: Duration::from_secs(3),
            memory_requirement: 200,
        },
        Task {
            id: 2,
            priority: 1,
            execution_time: Duration::from_secs(2),
            memory_requirement: 300,
        },
    ];

    for task in tasks {
        scheduler.add_task(task);
    }

    assert_eq!(scheduler.tasks.len(), 2);
    assert_eq!(memory_manager.get_available_memory(), 1000);

    memory_manager.allocate_for_tasks(&scheduler.tasks).unwrap();

    assert_eq!(memory_manager.get_available_memory(), 500);

    scheduler.schedule();

    assert_eq!(scheduler.tasks.len(), 0);
}

#[test]
fn test_power_management() {
    let mut power_manager = PowerManager::new();

    power_manager.optimize_power(0.2);
    assert!(matches!(
        power_manager.get_power_state(),
        PowerState::LowPower
    ));

    power_manager.optimize_power(0.5);
    assert!(matches!(
        power_manager.get_power_state(),
        PowerState::Normal
    ));

    power_manager.optimize_power(0.8);
    assert!(matches!(
        power_manager.get_power_state(),
        PowerState::HighPerformance
    ));
}

#[test]
fn test_integrated_system() {
    let num_processing_units = 4;
    let total_memory = 2000;
    let mut scheduler = TaskScheduler::new(num_processing_units);
    let mut memory_manager = MemoryManager::new(total_memory);
    let mut power_manager = PowerManager::new();

    let tasks = vec![
        Task {
            id: 1,
            priority: 3,
            execution_time: Duration::from_secs(5),
            memory_requirement: 300,
        },
        Task {
            id: 2,
            priority: 1,
            execution_time: Duration::from_secs(2),
            memory_requirement: 200,
        },
        Task {
            id: 3,
            priority: 2,
            execution_time: Duration::from_secs(4),
            memory_requirement: 400,
        },
    ];

    for task in tasks {
        scheduler.add_task(task);
    }

    memory_manager.allocate_for_tasks(&scheduler.tasks).unwrap();
    assert_eq!(memory_manager.get_available_memory(), 1100);

    scheduler.schedule();
    assert_eq!(scheduler.tasks.len(), 0);

    let system_load = 0.6;
    power_manager.optimize_power(system_load);
    assert!(matches!(
        power_manager.get_power_state(),
        PowerState::Normal
    ));
}

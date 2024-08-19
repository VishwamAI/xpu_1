mod memory_management;
mod power_management;
mod task_scheduling;

use memory_management::MemoryManager;
use power_management::PowerManager;
use std::time::Duration;
use task_scheduling::{Task, TaskScheduler, ProcessingUnitType};

fn main() {
    println!("XPU Manager Rust Implementation");

    // Initialize components
    let num_processing_units = 4; // Default value
    let total_memory = 1024; // Default value in bytes
    let mut scheduler = TaskScheduler::new(num_processing_units);
    let mut memory_manager = MemoryManager::new(total_memory);
    let mut power_manager = PowerManager::new();

    // Create some sample tasks
    let tasks = vec![
        Task {
            id: 1,
            priority: 2,
            execution_time: Duration::from_secs(5),
            memory_requirement: 200,
            unit_type: ProcessingUnitType::CPU,
        },
        Task {
            id: 2,
            priority: 1,
            execution_time: Duration::from_secs(3),
            memory_requirement: 100,
            unit_type: ProcessingUnitType::GPU,
        },
        Task {
            id: 3,
            priority: 3,
            execution_time: Duration::from_secs(7),
            memory_requirement: 300,
            unit_type: ProcessingUnitType::NPU,
        },
    ];

    // Add tasks to the scheduler
    for task in tasks {
        scheduler.add_task(task);
    }

    // Allocate memory for tasks
    if let Err(e) = memory_manager.allocate_for_tasks(&scheduler.tasks) {
        println!("Memory allocation error: {}", e);
        return;
    }

    // Schedule and execute tasks
    let completed_tasks = scheduler.schedule();

    // Deallocate memory for completed tasks
    if let Err(e) = memory_manager.deallocate_completed_tasks(&completed_tasks) {
        println!("Memory deallocation error: {}", e);
    }

    // Simulate system load and optimize power
    let system_load = 0.6; // 60% load for demonstration
    power_manager.optimize_power(system_load);

    println!("Current power state: {:?}", power_manager.get_power_state());
    println!(
        "Power consumption: {:.2} W",
        power_manager.get_power_consumption()
    );
    println!(
        "Available memory: {} bytes",
        memory_manager.get_available_memory()
    );
}

use xpu_manager_rust::memory_management::MemoryManager;
use xpu_manager_rust::power_management::{PowerManager, PowerState, EnergyProfile};
use xpu_manager_rust::task_scheduling::{Task, TaskScheduler, ProcessingUnitType, ProcessingUnit, OptimizationMetrics};
use xpu_manager_rust::adaptive_optimization::AdaptiveOptimizer;
use std::time::Duration;

fn main() {
    println!("XPU Manager Rust Implementation with Adaptive Optimization");

    // Initialize components
    let num_processing_units = 4; // Default value
    let total_memory = 10485760; // 10 MB in bytes
    let mut scheduler = TaskScheduler::new(num_processing_units);
    let mut memory_manager = MemoryManager::new(total_memory);
    let mut power_manager = PowerManager::new();
    let mut adaptive_optimizer = AdaptiveOptimizer::new();

    // Create some sample tasks
    let tasks = vec![
        Task {
            id: 1,
            priority: 2,
            execution_time: Duration::from_secs(5),
            memory_requirement: 2_097_152, // 2 MB
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
            estimated_duration: Duration::from_secs(6), // Estimated duration slightly higher than execution time
            estimated_resource_usage: 2_200_000, // Estimated resource usage slightly higher than memory requirement
        },
        Task {
            id: 2,
            priority: 1,
            execution_time: Duration::from_secs(3),
            memory_requirement: 1_048_576, // 1 MB
            unit_type: ProcessingUnitType::GPU,
            dependencies: vec![],
            secure: false,
            unit: ProcessingUnit {
                id: 1,
                unit_type: ProcessingUnitType::GPU,
                current_load: Duration::new(0, 0),
                processing_power: 1.0,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            },
            estimated_duration: Duration::from_secs(4), // Estimated duration slightly higher than execution time
            estimated_resource_usage: 1_100_000, // Estimated resource usage slightly higher than memory requirement
        },
        Task {
            id: 3,
            priority: 3,
            execution_time: Duration::from_secs(7),
            memory_requirement: 3_145_728, // 3 MB
            unit_type: ProcessingUnitType::NPU,
            dependencies: vec![],
            secure: false,
            unit: ProcessingUnit {
                id: 2,
                unit_type: ProcessingUnitType::NPU,
                current_load: Duration::new(0, 0),
                processing_power: 1.0,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            },
            estimated_duration: Duration::from_secs(8), // Estimated duration slightly higher than execution time
            estimated_resource_usage: 3_300_000, // Estimated resource usage slightly higher than memory requirement
        },
    ];

    // Add tasks to the scheduler
    for task in tasks {
        scheduler.add_task(task);
    }

    // Adaptive optimization loop
    for iteration in 1..=5 {
        println!("\nOptimization Iteration {}", iteration);

        // Allocate memory for tasks
        if let Err(e) = memory_manager.allocate_for_tasks(scheduler.tasks.make_contiguous()) {
            println!("Memory allocation error: {}", e);
            return;
        }

        // Schedule and execute tasks
        let (completed_tasks, metrics) = match scheduler.schedule_with_metrics() {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error during task scheduling: {}", e);
                (Vec::new(), OptimizationMetrics::default())
            }
        };

        // Deallocate memory for completed tasks
        if let Err(e) = memory_manager.deallocate_completed_tasks(&completed_tasks) {
            println!("Memory deallocation error: {}", e);
        }

        // Optimize power based on current system load
        let system_load = metrics.average_load;
        power_manager.optimize_power(system_load);

        // Perform adaptive optimization
        match adaptive_optimizer.optimize(&metrics) {
            Ok(optimization_params) => scheduler.apply_optimization(optimization_params),
            Err(e) => println!("Error during optimization: {:?}", e),
        }

        // Print current state
        println!("Tasks completed: {}", completed_tasks.len());
        println!("Average task latency: {:?}", metrics.average_latency);
        println!("Current power state: {:?}", power_manager.get_power_state());
        println!("Power consumption: {:.2} W", power_manager.get_power_consumption());
        println!("Available memory: {} bytes", memory_manager.get_available_memory());
        println!("Memory fragmentation: {:.2}%", memory_manager.get_fragmentation_percentage());
    }

    println!("\nXPU optimization completed.");
}

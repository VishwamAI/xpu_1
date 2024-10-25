use std::time::Duration;
use std::sync::{Arc, Mutex};
use xpu_manager_rust::{
    EnergyProfile, MemoryManager, PowerManager, PowerState, ProcessingUnitType,
    Task, Scheduler, XpuOptimizer, XpuOptimizerConfig, SchedulerType, MemoryManagerType,
    power_management::PowerManagementPolicy,
    cloud_offloading::CloudOffloadingPolicy,
};

mod test_helpers;
use test_helpers::{initialize_test_env, setup_test_user, cleanup_test_data};

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
            power_management_policy: PowerManagementPolicy::Default,
            cloud_offloading_policy: CloudOffloadingPolicy::Default,
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

        // Initialize test environment and set up authentication
        println!("Starting test environment initialization...");
        initialize_test_env(&mut optimizer)?;
        println!("Test environment initialized successfully");

        println!("Setting up test user and generating token...");
        let token = setup_test_user(&mut optimizer)?;
        println!("Generated token: {}", token);

        println!("Setting current token...");
        match optimizer.set_current_token(token.clone()) {
            Ok(_) => println!("Token set successfully"),
            Err(e) => {
                println!("Failed to set token: {:?}", e);
                return Err(Box::new(e));
            }
        };

        let task = Task::new(
            1,
            1,
            vec![],
            Duration::from_secs(1),
            100,
            false,
            ProcessingUnitType::CPU,
        );

        optimizer.add_task(task, &token)?;
        assert_eq!(optimizer.task_queue.len(), 1);

        // Clean up test data
        println!("Cleaning up test data...");
        cleanup_test_data(&mut optimizer)?;
        println!("Test completed successfully");
        Ok(())
    }

    #[test]
    fn test_schedule_tasks() -> Result<(), Box<dyn std::error::Error>> {
        let config = XpuOptimizerConfig {
            num_processing_units: 4,
            ..Default::default()
        };
        let mut optimizer = XpuOptimizer::new(config)?;

        // Initialize test environment and set up authentication
        println!("Starting test environment initialization...");
        initialize_test_env(&mut optimizer)?;
        println!("Test environment initialized successfully");

        println!("Setting up test user and generating token...");
        let token = setup_test_user(&mut optimizer)?;
        println!("Generated token: {}", token);

        println!("Setting current token...");
        match optimizer.set_current_token(token.clone()) {
            Ok(_) => println!("Token set successfully"),
            Err(e) => {
                println!("Failed to set token: {:?}", e);
                return Err(Box::new(e));
            }
        };

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
            optimizer.add_task(task, &token)?;
        }

        assert_eq!(optimizer.task_queue.len(), 5);
        optimizer.run()?;
        assert!(optimizer.task_queue.is_empty());

        // Clean up test data
        println!("Cleaning up test data...");
        cleanup_test_data(&mut optimizer)?;
        println!("Test completed successfully");
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

        // Initialize test environment and set up authentication
        println!("Starting test environment initialization...");
        initialize_test_env(&mut optimizer)?;
        println!("Test environment initialized successfully");

        println!("Setting up test user and generating token...");
        let token = setup_test_user(&mut optimizer)?;
        println!("Generated token: {}", token);

        println!("Setting current token...");
        match optimizer.set_current_token(token.clone()) {
            Ok(_) => println!("Token set successfully"),
            Err(e) => {
                println!("Failed to set token: {:?}", e);
                return Err(Box::new(e));
            }
        };

        // Add tasks with memory requirements
        println!("Adding tasks with authenticated token...");
        let mut added_tasks = Vec::new();
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
            println!("Adding task {} with token", i);
            match optimizer.add_task(task.clone(), &token) {
                Ok(_) => {
                    println!("Successfully added task {}", i);
                    added_tasks.push(task);
                },
                Err(e) => {
                    println!("Failed to add task {}: {:?}", i, e);
                    return Err(Box::new(e));
                }
            }
        }

        println!("Verifying task queue contents...");
        assert_eq!(optimizer.task_queue.len(), 5, "Expected 5 tasks in queue");

        println!("Collecting tasks for memory allocation...");
        let tasks: Vec<Task> = optimizer.task_queue.iter().cloned().collect();
        println!("Allocating memory for {} tasks...", tasks.len());
        match optimizer.allocate_memory_for_tasks(&tasks) {
            Ok(_) => println!("Successfully allocated memory for tasks"),
            Err(e) => {
                println!("Failed to allocate memory: {:?}", e);
                return Err(Box::new(e));
            }
        }

        // Create a new scope for the memory manager lock
        {
            let memory_manager = optimizer.memory_manager.lock().unwrap();
            let available_memory = memory_manager.get_available_memory();
            println!("Current available memory: {}", available_memory);
            assert_eq!(available_memory, 1024 - 5 * 100,
                "Expected {} memory available, got {}", 1024 - 5 * 100, available_memory);
        } // memory_manager lock is dropped here

        // Clean up test data
        println!("Cleaning up test data...");
        cleanup_test_data(&mut optimizer)?;
        println!("Test completed successfully");
        Ok(())
    }
}

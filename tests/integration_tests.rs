use std::time::Duration;
use std::sync::{Arc, Mutex};
use xpu_manager_rust::{
    memory_management::{MemoryManager, SimpleMemoryManager},
    power_management::{PowerManager, PowerState, PowerManagementPolicy},
    task_scheduling::{ProcessingUnitType, Task, Scheduler, SchedulerType},
    XpuOptimizerError,
    xpu_optimization::{XpuOptimizer, XpuOptimizerConfig, UserRole},
    cloud_offloading::CloudOffloadingPolicy,
};
mod test_helpers;

#[test]
fn test_task_scheduling_and_memory_allocation() -> Result<(), XpuOptimizerError> {
    let config = XpuOptimizerConfig {
        num_processing_units: 2,
        memory_pool_size: 1000,
        scheduler_type: SchedulerType::RoundRobin,
        memory_manager_type: xpu_manager_rust::memory_management::MemoryManagerType::Simple,
        power_management_policy: PowerManagementPolicy::Default,
        cloud_offloading_policy: CloudOffloadingPolicy::Default,
        adaptive_optimization_policy: "default".to_string(),
    };
    let mut optimizer = XpuOptimizer::new(config)?;

    // Initialize test environment and get admin token for task operations
    let admin_token = test_helpers::create_test_session(&mut optimizer, UserRole::Admin)?;

    let tasks = vec![
        Task::new(
            1,
            2,
            vec![],
            Duration::from_secs(3),
            200,
            false,
            ProcessingUnitType::CPU,
        ),
        Task::new(
            2,
            1,
            vec![],
            Duration::from_secs(2),
            300,
            false,
            ProcessingUnitType::GPU,
        ),
    ];

    for task in &tasks {
        optimizer.add_task(task.clone(), &admin_token)?;
    }

    assert_eq!(optimizer.task_queue.len(), 2);

    let memory_manager = optimizer.memory_manager.lock().map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?;
    assert_eq!(memory_manager.get_available_memory(), 1000);
    drop(memory_manager);

    optimizer.run()?;

    let memory_manager = optimizer.memory_manager.lock().map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?;
    assert_eq!(memory_manager.get_available_memory(), 500);
    drop(memory_manager);

    assert_eq!(optimizer.task_queue.len(), 0);

    Ok(())
}

#[test]
fn test_power_management() -> Result<(), XpuOptimizerError> {
    let mut power_manager = PowerManager::new();

    power_manager.optimize_power(0.2)?;
    assert!(matches!(power_manager.get_power_state(), PowerState::LowPower));

    power_manager.optimize_power(0.5)?;
    assert!(matches!(power_manager.get_power_state(), PowerState::Normal));

    power_manager.optimize_power(0.8)?;
    assert!(matches!(power_manager.get_power_state(), PowerState::HighPerformance));

    Ok(())
}

#[test]
fn test_integrated_system() -> Result<(), XpuOptimizerError> {
    let config = XpuOptimizerConfig {
        num_processing_units: 4,
        memory_pool_size: 2000,
        scheduler_type: SchedulerType::RoundRobin,
        memory_manager_type: xpu_manager_rust::memory_management::MemoryManagerType::Simple,
        power_management_policy: PowerManagementPolicy::Default,
        cloud_offloading_policy: CloudOffloadingPolicy::Default,
        adaptive_optimization_policy: "default".to_string(),
    };
    let mut optimizer = XpuOptimizer::new(config)?;

    // Initialize test environment and get admin token for task operations
    let admin_token = test_helpers::create_test_session(&mut optimizer, UserRole::Admin)?;

    let tasks = vec![
        Task::new(
            1,
            3,
            vec![],
            Duration::from_secs(5),
            300,
            false,
            ProcessingUnitType::CPU,
        ),
        Task::new(
            2,
            1,
            vec![],
            Duration::from_secs(2),
            200,
            false,
            ProcessingUnitType::GPU,
        ),
        Task::new(
            3,
            2,
            vec![],
            Duration::from_secs(4),
            400,
            false,
            ProcessingUnitType::NPU,
        ),
    ];

    for task in tasks {
        optimizer.add_task(task, &admin_token)?;
    }

    let memory_manager = optimizer.memory_manager.lock().map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?;
    assert_eq!(memory_manager.get_available_memory(), 2000);
    drop(memory_manager);

    optimizer.run()?;

    let memory_manager = optimizer.memory_manager.lock().map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?;
    assert_eq!(memory_manager.get_available_memory(), 1100);
    drop(memory_manager);

    assert_eq!(optimizer.task_queue.len(), 0);

    let system_load = 0.6;
    optimizer.power_manager.optimize_power(system_load)?;
    assert!(matches!(optimizer.power_manager.get_power_state(), PowerState::Normal));

    Ok(())
}

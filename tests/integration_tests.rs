use std::time::Duration;
use itertools::Itertools;
use xpu_manager_rust::{
    power_management::{PowerManager, PowerState, PowerManagementPolicy},
    task_scheduling::{ProcessingUnitType, Task, SchedulerType},
    XpuOptimizerError,
    xpu_optimization::{XpuOptimizer, XpuOptimizerConfig},
    cloud_offloading::CloudOffloadingPolicy,
    task_data::{HistoricalTaskData, TaskExecutionData},
};

fn create_test_user_and_authenticate(optimizer: &mut XpuOptimizer) -> Result<String, XpuOptimizerError> {
    optimizer.add_user("test_user".to_string(), "test_password".to_string(), xpu_manager_rust::xpu_optimization::UserRole::User)?;
    optimizer.authenticate_user("test_user", "test_password")
}

fn create_historical_data() -> Vec<TaskExecutionData> {
    vec![
        TaskExecutionData {
            id: 1,
            execution_time: Duration::from_secs(2),
            memory_usage: 150,
            unit_type: ProcessingUnitType::CPU,
            priority: 2,
            success: true,
            memory_requirement: 150,
        },
        TaskExecutionData {
            id: 2,
            execution_time: Duration::from_secs(3),
            memory_usage: 250,
            unit_type: ProcessingUnitType::GPU,
            priority: 1,
            success: true,
            memory_requirement: 250,
        },
    ]
}

#[test]
fn test_task_scheduling_and_memory_allocation() -> Result<(), XpuOptimizerError> {
    let config = XpuOptimizerConfig {
        num_processing_units: 7,
        memory_pool_size: 2000,
        scheduler_type: SchedulerType::RoundRobin,
        memory_manager_type: xpu_manager_rust::memory_management::MemoryManagerType::Dynamic,
        power_management_policy: PowerManagementPolicy::Default,
        cloud_offloading_policy: CloudOffloadingPolicy::Default,
        adaptive_optimization_policy: "default".to_string(),
    };
    let mut optimizer = XpuOptimizer::new(config.clone())?;

    let historical_data = create_historical_data();
    optimizer.task_history = historical_data;

    let token = create_test_user_and_authenticate(&mut optimizer)?;

    let tasks = vec![
        Task::new(1, 7, vec![], Duration::from_secs(3), 200, false, ProcessingUnitType::CPU),
        Task::new(2, 6, vec![], Duration::from_secs(2), 300, false, ProcessingUnitType::GPU),
        Task::new(3, 5, vec![], Duration::from_secs(4), 400, false, ProcessingUnitType::TPU),
        Task::new(4, 4, vec![], Duration::from_secs(2), 100, false, ProcessingUnitType::NPU),
        Task::new(5, 3, vec![], Duration::from_secs(1), 150, false, ProcessingUnitType::LPU),
        Task::new(6, 2, vec![], Duration::from_secs(3), 250, false, ProcessingUnitType::VPU),
        Task::new(7, 1, vec![], Duration::from_secs(5), 350, false, ProcessingUnitType::FPGA),
    ];

    let total_memory_requirement: usize = tasks.iter().map(|t| t.memory_requirement).sum();
    assert!(total_memory_requirement <= config.memory_pool_size, "Total memory requirement should not exceed memory pool size");

    for task in &tasks {
        optimizer.add_task(task.clone(), &token)?;
    }

    assert_eq!(optimizer.task_queue.len(), 7, "All tasks should be initially queued");

    let initial_memory = optimizer.get_available_memory()?;
    assert_eq!(initial_memory, config.memory_pool_size, "Initial available memory should match the configured memory pool size");

    let run_result = optimizer.run();
    assert!(run_result.is_ok(), "Optimizer run should succeed: {:?}", run_result);

    let final_memory = optimizer.get_available_memory()?;
    assert_eq!(final_memory, initial_memory, "All memory should be deallocated after execution");

    assert!(optimizer.task_queue.is_empty(), "All tasks should have been processed");
    assert_eq!(optimizer.scheduled_tasks.len(), 7, "All tasks should have been scheduled");

    // Verify task execution
    let executed_tasks: Vec<_> = optimizer.task_history.iter()
        .filter(|t| tasks.iter().any(|task| task.id == t.id))
        .collect();
    assert_eq!(executed_tasks.len(), 7, "All tasks should have been executed");

    // Check if tasks were executed in priority order
    let executed_task_ids: Vec<usize> = executed_tasks.iter().map(|t| t.id).collect();
    assert!(is_sorted_by_key(&executed_task_ids, |&id| std::cmp::Reverse(id)), "Tasks should be executed in priority order");

    // Verify that each task was executed on the correct processing unit type
    for task in &tasks {
        let executed_task = executed_tasks.iter().find(|t| t.id == task.id).unwrap();
        assert_eq!(executed_task.unit_type, task.unit_type, "Task {} should be executed on {:?}", task.id, task.unit_type);
        assert_eq!(executed_task.memory_usage, task.memory_requirement, "Task {} should use the correct amount of memory", task.id);
    }

    // Verify memory allocation and deallocation
    let memory_usage_timeline = optimizer.get_memory_usage_timeline()?;
    assert!(!memory_usage_timeline.is_empty(), "Memory usage timeline should not be empty");
    assert!(memory_usage_timeline.iter().all(|point| point.used_memory <= point.total_memory),
            "Used memory should never exceed the total memory");

    // Check for proper memory allocation and deallocation
    assert_eq!(memory_usage_timeline[0].used_memory, 0, "Initial memory usage should be 0");
    assert_eq!(memory_usage_timeline.last().unwrap().used_memory, 0, "Final memory usage should be 0");

    // Verify that memory was allocated and deallocated correctly
    let max_memory_usage = memory_usage_timeline.iter().map(|point| point.used_memory).max().unwrap_or(0);
    assert!(max_memory_usage > 0, "Maximum memory usage should be greater than 0");
    assert!(max_memory_usage <= total_memory_requirement, "Maximum memory usage should not exceed total memory requirement");

    // Check for memory usage patterns
    let peak_usage_count = memory_usage_timeline.iter().filter(|point| point.used_memory == max_memory_usage).count();
    assert!(peak_usage_count > 0, "There should be at least one peak memory usage point");
    assert!(memory_usage_timeline.windows(2).any(|w| w[0].used_memory < w[1].used_memory), "Memory usage should increase at some point");
    assert!(memory_usage_timeline.windows(2).any(|w| w[0].used_memory > w[1].used_memory), "Memory usage should decrease at some point");

    // Verify total memory remains constant
    assert!(memory_usage_timeline.iter().all(|point| point.total_memory == config.memory_pool_size),
            "Total memory should remain constant and equal to the configured memory pool size");

    // Test edge case: Add a task that requires more memory than available
    let large_task = Task::new(8, 8, vec![], Duration::from_secs(10), 2500, false, ProcessingUnitType::CPU);
    let result = optimizer.add_task(large_task, &token);
    assert!(matches!(result, Err(XpuOptimizerError::MemoryError(_))), "Adding a task with excessive memory requirements should fail");

    // Test concurrent execution
    let concurrent_tasks = vec![
        Task::new(9, 5, vec![], Duration::from_secs(2), 100, false, ProcessingUnitType::CPU),
        Task::new(10, 5, vec![], Duration::from_secs(2), 100, false, ProcessingUnitType::GPU),
    ];

    for task in &concurrent_tasks {
        optimizer.add_task(task.clone(), &token)?;
    }

    let run_result = optimizer.run();
    assert!(run_result.is_ok(), "Concurrent task execution should succeed");

    let executed_concurrent_tasks: Vec<_> = optimizer.task_history.iter()
        .filter(|t| concurrent_tasks.iter().any(|task| task.id == t.id))
        .collect();
    assert_eq!(executed_concurrent_tasks.len(), 2, "Both concurrent tasks should have been executed");

    Ok(())
}

// Custom sorting check function
fn is_sorted_by_key<T, F, K>(slice: &[T], mut f: F) -> bool
where
    F: FnMut(&T) -> K,
    K: Ord,
{
    slice.windows(2).all(|w| f(&w[0]) <= f(&w[1]))
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
        num_processing_units: 7,
        memory_pool_size: 2000,
        scheduler_type: SchedulerType::LoadBalancing,
        memory_manager_type: xpu_manager_rust::memory_management::MemoryManagerType::Dynamic,
        power_management_policy: PowerManagementPolicy::Aggressive,
        cloud_offloading_policy: CloudOffloadingPolicy::Always,
        adaptive_optimization_policy: "ml-driven".to_string(),
    };
    let mut optimizer = XpuOptimizer::new(config)?;

    // Add historical data for ML optimization
    let historical_data = create_historical_data();
    optimizer.task_history = historical_data.clone();

    let token = create_test_user_and_authenticate(&mut optimizer)?;

    let tasks = vec![
        Task::new(1, 3, vec![], Duration::from_secs(5), 200, false, ProcessingUnitType::CPU),
        Task::new(2, 1, vec![], Duration::from_secs(2), 150, false, ProcessingUnitType::GPU),
        Task::new(3, 2, vec![], Duration::from_secs(4), 300, false, ProcessingUnitType::NPU),
        Task::new(4, 4, vec![], Duration::from_secs(3), 100, false, ProcessingUnitType::TPU),
        Task::new(5, 2, vec![], Duration::from_secs(1), 50, false, ProcessingUnitType::LPU),
        Task::new(6, 3, vec![], Duration::from_secs(2), 200, false, ProcessingUnitType::VPU),
        Task::new(7, 1, vec![], Duration::from_secs(3), 250, false, ProcessingUnitType::FPGA),
        Task::new(8, 5, vec![], Duration::from_secs(6), 400, false, ProcessingUnitType::CPU),
        // Edge case: Task that requires more memory than available
        Task::new(9, 6, vec![], Duration::from_secs(10), 2500, false, ProcessingUnitType::CPU),
    ];

    for task in &tasks {
        optimizer.add_task(task.clone(), &token)?;
    }

    let initial_memory = optimizer.get_available_memory()?;
    assert_eq!(initial_memory, 2000, "Initial memory should be 2000");

    let run_result = optimizer.run();
    assert!(matches!(run_result, Err(XpuOptimizerError::TaskExecutionError(_))),
            "Expected TaskExecutionError due to insufficient memory");

    let final_memory = optimizer.get_available_memory()?;
    assert!(final_memory > 0, "Some memory should be available after task execution");
    assert!(final_memory < 2000, "Some memory should have been used");

    assert!(!optimizer.task_queue.is_empty(), "Some tasks should remain unprocessed due to memory constraints");

    // Check if some tasks were executed
    let executed_tasks = optimizer.task_history.len() - historical_data.len();
    assert!(executed_tasks > 0 && executed_tasks < tasks.len(),
            "Some tasks should be in task history, but not all. Executed: {}", executed_tasks);

    // Verify that tasks were executed in priority order, respecting memory constraints
    let executed_task_ids: Vec<usize> = optimizer.task_history.iter().skip(historical_data.len()).map(|t| t.id).collect();
    let expected_order: Vec<usize> = tasks.iter()
        .filter(|t| t.memory_requirement <= 2000)
        .sorted_by_key(|t| (std::cmp::Reverse(t.priority), t.id))
        .map(|t| t.id)
        .collect();
    assert_eq!(executed_task_ids, expected_order,
               "Tasks should be executed in priority order, respecting memory constraints");

    // System load and power management assertions
    let power_state = optimizer.power_manager.get_power_state();
    assert!(matches!(power_state, PowerState::HighPerformance),
            "Power state should be HighPerformance due to Aggressive policy");

    // Test adaptive optimization
    let adaptive_result = optimizer.run();
    assert!(matches!(adaptive_result, Err(XpuOptimizerError::TaskExecutionError(_))),
            "Expected TaskExecutionError in adaptive optimization");

    // Verify that the task history has been updated
    assert!(optimizer.task_history.len() > historical_data.len(),
            "Task history should be populated after running the optimizer. Current: {}, Initial: {}",
            optimizer.task_history.len(), historical_data.len());

    // Test error handling for invalid task
    let invalid_task = Task::new(10, 1, vec![], Duration::from_secs(1), 100, false, ProcessingUnitType::CPU);
    let invalid_result = optimizer.add_task(invalid_task, "invalid_token");
    assert!(matches!(invalid_result, Err(XpuOptimizerError::AuthenticationError(_))),
            "Adding task with invalid token should result in AuthenticationError");

    // Test cloud offloading
    let cloud_offloadable_task = Task::new(11, 1, vec![], Duration::from_secs(2), 3000, false, ProcessingUnitType::CPU);
    optimizer.add_task(cloud_offloadable_task.clone(), &token)?;
    let cloud_result = optimizer.run();
    assert!(matches!(cloud_result, Ok(())), "Task should be offloaded to cloud successfully");

    // Verify cloud offloading
    let cloud_task_executed = optimizer.task_history.iter().any(|t| t.id == cloud_offloadable_task.id);
    assert!(cloud_task_executed, "Cloud offloadable task should be in task history");

    Ok(())
}

#[test]
fn test_token_expiration() -> Result<(), XpuOptimizerError> {
    let config = XpuOptimizerConfig::default();
    let mut optimizer = XpuOptimizer::new(config)?;

    let token = create_test_user_and_authenticate(&mut optimizer)?;

    // Simulate token expiration
    optimizer.sessions.clear();

    let task = Task::new(
        1,
        1,
        vec![],
        Duration::from_secs(1),
        100,
        false,
        ProcessingUnitType::CPU,
    );

    let result = optimizer.add_task(task, &token);
    assert!(matches!(result, Err(XpuOptimizerError::SessionNotFoundError)));

    Ok(())
}

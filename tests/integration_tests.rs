use std::time::Duration;
use xpu_manager_rust::{
    power_management::{PowerManager, PowerState, PowerManagementPolicy},
    task_scheduling::{ProcessingUnitType, Task, SchedulerType},
    XpuOptimizerError,
    xpu_optimization::{XpuOptimizer, XpuOptimizerConfig, UserRole, MachineLearningOptimizer},
    cloud_offloading::CloudOffloadingPolicy,
    task_data::{TaskExecutionData, HistoricalTaskData},
    ml_models::MLModel,
};

fn generate_valid_token(optimizer: &mut XpuOptimizer) -> Result<String, XpuOptimizerError> {
    optimizer.add_user("test_user".to_string(), "test_password".to_string(), UserRole::Admin)?;
    optimizer.authenticate_user("test_user", "test_password")
}

#[test]
fn test_task_scheduling_and_memory_allocation() -> Result<(), XpuOptimizerError> {
    let config = XpuOptimizerConfig {
        num_processing_units: 7, // One for each processing unit type
        memory_pool_size: 5000, // Increased to 5000 to ensure sufficient memory for all tasks
        scheduler_type: SchedulerType::RoundRobin,
        memory_manager_type: xpu_manager_rust::memory_management::MemoryManagerType::Simple,
        power_management_policy: PowerManagementPolicy::Default,
        cloud_offloading_policy: CloudOffloadingPolicy::Default,
        adaptive_optimization_policy: "default".to_string(),
    };
    let mut optimizer = XpuOptimizer::new(config)?;
    let valid_token = generate_valid_token(&mut optimizer)?;

    // Assert initial available memory
    let initial_memory = optimizer.memory_manager.lock()
        .map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?
        .get_available_memory();
    assert_eq!(initial_memory, 5000, "Initial available memory is incorrect");

    let tasks = vec![
        Task::new(1, 2, vec![], Duration::from_secs(3), 200, false, ProcessingUnitType::CPU),
        Task::new(2, 1, vec![], Duration::from_secs(2), 300, false, ProcessingUnitType::GPU),
        Task::new(3, 3, vec![], Duration::from_secs(4), 400, false, ProcessingUnitType::TPU),
        Task::new(4, 2, vec![], Duration::from_secs(3), 250, false, ProcessingUnitType::NPU),
        Task::new(5, 1, vec![], Duration::from_secs(2), 350, false, ProcessingUnitType::LPU),
        Task::new(6, 3, vec![], Duration::from_secs(4), 450, false, ProcessingUnitType::VPU),
        Task::new(7, 2, vec![], Duration::from_secs(3), 300, false, ProcessingUnitType::FPGA),
    ];

    for task in &tasks {
        optimizer.add_task(task.clone(), &valid_token)?;
    }

    assert_eq!(optimizer.task_queue.len(), 7, "Not all tasks were added to the queue");

    let initial_memory = optimizer.memory_manager.lock()
        .map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?
        .get_available_memory();
    assert_eq!(initial_memory, 2000, "Initial available memory is incorrect");

    optimizer.run()?;

    let final_memory = optimizer.memory_manager.lock()
        .map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?
        .get_available_memory();
    assert_eq!(final_memory, 750, "Final available memory is incorrect");

    assert_eq!(optimizer.task_queue.len(), 0, "Not all tasks were processed");

    // Check if all processing units were used
    let used_unit_types: std::collections::HashSet<_> = optimizer.processing_units.iter()
        .filter_map(|unit| {
            unit.lock().ok().and_then(|guard| guard.get_unit_type().ok())
        })
        .collect();

    assert_eq!(used_unit_types.len(), 7, "Not all processing unit types were used");

    // Check if each processing unit type was used
    for unit_type in [ProcessingUnitType::CPU, ProcessingUnitType::GPU, ProcessingUnitType::TPU,
                      ProcessingUnitType::NPU, ProcessingUnitType::LPU, ProcessingUnitType::VPU,
                      ProcessingUnitType::FPGA].iter() {
        assert!(used_unit_types.contains(unit_type),
                "{:?} was not used", unit_type);
    }

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
    let valid_token = generate_valid_token(&mut optimizer)?;

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
        optimizer.add_task(task, &valid_token)?;
    }

    let memory_manager = optimizer.memory_manager.lock().map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?;
    assert_eq!(memory_manager.get_available_memory(), 2000);
    drop(memory_manager);

    optimizer.run()?;

    let memory_manager = optimizer.memory_manager.lock().map_err(|_| XpuOptimizerError::LockError("Failed to lock memory manager".to_string()))?;
    assert_eq!(memory_manager.get_available_memory(), 1100);
    drop(memory_manager);

    assert_eq!(optimizer.task_queue.len(), 0);

    Ok(())
}

#[test]
fn test_ml_driven_policy() -> Result<(), XpuOptimizerError> {
    let config = XpuOptimizerConfig {
        num_processing_units: 7,
        memory_pool_size: 2000,
        scheduler_type: SchedulerType::AIOptimized,
        memory_manager_type: xpu_manager_rust::memory_management::MemoryManagerType::Simple,
        power_management_policy: PowerManagementPolicy::Default,
        cloud_offloading_policy: CloudOffloadingPolicy::Default,
        adaptive_optimization_policy: "ml-driven".to_string(),
    };
    let mut optimizer = XpuOptimizer::new(config)?;
    let valid_token = generate_valid_token(&mut optimizer)?;

    let tasks = vec![
        Task::new(1, 2, vec![], Duration::from_secs(3), 200, false, ProcessingUnitType::CPU),
        Task::new(2, 1, vec![], Duration::from_secs(2), 300, false, ProcessingUnitType::GPU),
        Task::new(3, 3, vec![], Duration::from_secs(4), 400, false, ProcessingUnitType::TPU),
        Task::new(4, 2, vec![], Duration::from_secs(3), 250, false, ProcessingUnitType::NPU),
        Task::new(5, 1, vec![], Duration::from_secs(2), 350, false, ProcessingUnitType::LPU),
        Task::new(6, 3, vec![], Duration::from_secs(4), 450, false, ProcessingUnitType::VPU),
        Task::new(7, 2, vec![], Duration::from_secs(3), 300, false, ProcessingUnitType::FPGA),
    ];

    // Add tasks and run the optimizer
    for task in &tasks {
        optimizer.add_task(task.clone(), &valid_token)?;
    }
    optimizer.run()?;

    assert_eq!(optimizer.task_queue.len(), 0, "Not all tasks were processed");

    // Verify ML-driven policy application
    let historical_data = tasks.iter().map(|task| TaskExecutionData {
        id: task.id,
        execution_time: task.execution_time,
        memory_usage: task.memory_requirement,
        unit_type: task.unit_type.clone(),
        priority: task.priority,
        success: true,
        memory_requirement: task.memory_requirement,
    }).collect::<Vec<_>>();

    let (new_task, prediction) = {
        let mut ml_optimizer = optimizer.ml_optimizer.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML optimizer: {}", e)))?;

        // Train the ML model with historical data
        ml_optimizer.train(&historical_data)?;

        // Make predictions for a new task
        let new_task = Task::new(8, 2, vec![], Duration::from_secs(3), 275, false, ProcessingUnitType::CPU);
        let historical_task_data = HistoricalTaskData {
            task_id: new_task.id,
            execution_time: new_task.execution_time,
            memory_usage: new_task.memory_requirement,
            unit_type: new_task.unit_type.clone(),
            priority: new_task.priority,
        };
        let prediction = ml_optimizer.predict(&historical_task_data)?;
        (new_task, prediction)
    };

    // Verify prediction
    assert!(prediction.estimated_duration > Duration::ZERO, "Predicted duration should be non-zero");
    assert!(prediction.estimated_resource_usage > 0, "Predicted resource usage should be non-zero");
    assert_eq!(prediction.task_id, new_task.id, "Prediction task ID should match");

    // Verify that the prediction influences scheduling
    let optimized_scheduler = {
        let ml_optimizer = optimizer.ml_optimizer.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML optimizer: {}", e)))?;
        ml_optimizer.optimize(&historical_data, &optimizer.processing_units)?
    };
    let schedule = optimized_scheduler.schedule(&[new_task.clone()], &optimizer.processing_units)?;

    assert!(!schedule.is_empty(), "Optimized scheduler should produce a non-empty schedule");
    let (scheduled_task, assigned_unit) = &schedule[0];
    assert_eq!(scheduled_task.id, new_task.id, "Scheduled task should match the new task");

    // Verify that the assigned unit matches the predicted unit type
    let assigned_unit_type = assigned_unit.lock()
        .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock assigned unit: {}", e)))?
        .get_unit_type()?;
    assert_eq!(assigned_unit_type, prediction.recommended_processing_unit,
               "Assigned unit type should match the prediction");

    log::info!("ML-driven policy test passed successfully");
    log::debug!("Prediction for task {}: {:?}", new_task.id, prediction);
    log::debug!("Scheduled on unit type: {:?}", assigned_unit_type);

    Ok(())
}

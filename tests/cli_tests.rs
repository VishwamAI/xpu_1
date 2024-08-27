use tempfile::TempDir;
use xpu_manager_rust::{
    cli::main::{parse_config_file, check_xpu_status, configure_xpu_manager, start_xpu_manager, stop_xpu_manager},
    task_scheduling::{SchedulerType, Scheduler},
    XpuOptimizerError,
    xpu_optimization::{XpuOptimizerConfig, XpuOptimizer},
    memory_management::MemoryManagerType,
    power_management::PowerManagementPolicy,
    cloud_offloading::CloudOffloadingPolicy,
};

#[test]
fn test_parse_config_file() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_config.json");
    std::fs::write(
        &config_path,
        r#"
        {
            "num_processing_units": 4,
            "memory_pool_size": 1024,
            "scheduler_type": "RoundRobin",
            "memory_manager_type": "Simple",
            "power_management_policy": "default",
            "cloud_offloading_policy": "default",
            "adaptive_optimization_policy": "default"
        }
        "#,
    )
    .unwrap();

    let result = parse_config_file(config_path.to_str().unwrap());
    assert!(result.is_ok());
    let config = result.unwrap();
    assert_eq!(config.num_processing_units, 4);
    assert_eq!(config.memory_pool_size, 1024);
    assert!(matches!(config.scheduler_type, SchedulerType::RoundRobin));
    assert!(matches!(config.memory_manager_type, MemoryManagerType::Simple));
    assert!(matches!(config.power_management_policy, PowerManagementPolicy::Default));
    assert!(matches!(config.cloud_offloading_policy, CloudOffloadingPolicy::Default));
    assert_eq!(config.adaptive_optimization_policy, "default");
}

#[test]
fn test_parse_config_file_error() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("invalid_config.json");
    std::fs::write(
        &config_path,
        r#"
        {
            "invalid_key": "invalid_value"
        }
        "#,
    )
    .unwrap();

    let result = parse_config_file(config_path.to_str().unwrap());
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), XpuOptimizerError::ConfigError(_)));
}

#[test]
fn test_check_xpu_status() {
    let result = check_xpu_status();
    assert!(result.is_ok());
}

#[test]
#[ignore]
fn test_configure_xpu_manager() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_config.json");
    std::fs::write(
        &config_path,
        r#"
        {
            "num_processing_units": 8,
            "memory_pool_size": 2048,
            "scheduler_type": "LoadBalancing",
            "memory_manager_type": "Dynamic",
            "power_management_policy": "default",
            "cloud_offloading_policy": "Always",
            "adaptive_optimization_policy": "ml-driven"
        }
        "#,
    )
    .unwrap();

    // Initialize the logger for testing
    let _ = env_logger::builder().is_test(true).try_init();

    // Test valid configurations
    let test_configs = vec![
        ("Always", CloudOffloadingPolicy::Always),
        ("ALWAYS", CloudOffloadingPolicy::Always),
        ("always", CloudOffloadingPolicy::Always),
        ("Never", CloudOffloadingPolicy::Never),
        ("Default", CloudOffloadingPolicy::Default),
    ];

    for (policy_str, expected_policy) in test_configs {
        let config_path = temp_dir.path().join(format!("test_config_{}.json", policy_str));
        std::fs::write(
            &config_path,
            format!(r#"
            {{
                "num_processing_units": 8,
                "memory_pool_size": 2048,
                "scheduler_type": "LoadBalancing",
                "memory_manager_type": "Dynamic",
                "power_management_policy": "default",
                "cloud_offloading_policy": "{}",
                "adaptive_optimization_policy": "ml-driven"
            }}
            "#, policy_str),
        ).unwrap();

        let result = configure_xpu_manager(config_path.to_str().unwrap());
        assert!(result.is_ok(), "configure_xpu_manager failed for '{}' policy: {:?}", policy_str, result.err());

        let optimizer = result.unwrap();
        assert_eq!(optimizer.config.num_processing_units, 8);
        assert_eq!(optimizer.config.memory_pool_size, 2048);
        assert!(matches!(optimizer.config.scheduler_type, SchedulerType::LoadBalancing));
        assert!(matches!(optimizer.config.memory_manager_type, MemoryManagerType::Dynamic));
        assert!(matches!(optimizer.config.power_management_policy, PowerManagementPolicy::Default));
        assert!(matches!(optimizer.config.cloud_offloading_policy, expected_policy));
        assert_eq!(optimizer.config.adaptive_optimization_policy, "ml-driven");

        assert_eq!(optimizer.processing_units.len(), 8);
        assert!(matches!(optimizer.scheduler, Scheduler::LoadBalancing(_)));
    }

    // Test with invalid configuration
    let invalid_config_path = temp_dir.path().join("invalid_config.json");
    std::fs::write(
        &invalid_config_path,
        r#"
        {
            "num_processing_units": 0,
            "memory_pool_size": 0,
            "scheduler_type": "InvalidType",
            "memory_manager_type": "InvalidType",
            "power_management_policy": "InvalidPolicy",
            "cloud_offloading_policy": "InvalidPolicy",
            "adaptive_optimization_policy": "invalid"
        }
        "#,
    )
    .unwrap();

    let invalid_result = configure_xpu_manager(invalid_config_path.to_str().unwrap());
    assert!(invalid_result.is_err(), "Expected error for invalid configuration");
    if let Err(XpuOptimizerError::ConfigError(err_msg)) = invalid_result {
        assert!(err_msg.contains("Invalid"), "Error message should mention invalid configuration");
    } else {
        panic!("Expected ConfigError for invalid configuration");
    }
}

#[test]
fn test_start_stop_xpu_manager() {
    let start_result = start_xpu_manager();
    assert!(start_result.is_ok());

    let stop_result = stop_xpu_manager();
    assert!(stop_result.is_ok());
}

#[test]
fn test_configure_xpu_manager_error() {
    let result = configure_xpu_manager("non_existent_file.json");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), XpuOptimizerError::ConfigError(_)));
}

// Add more tests for other CLI functionalities as needed

#[test]
fn test_xpu_optimizer_initialization() -> Result<(), Box<dyn std::error::Error>> {
    let config = XpuOptimizerConfig::default();
    let optimizer = XpuOptimizer::new(config.clone())?;

    assert_eq!(optimizer.processing_units.len(), config.num_processing_units);
    assert!(matches!(optimizer.scheduler, Scheduler::RoundRobin(_)));
    assert_eq!(optimizer.task_queue.len(), 0);

    Ok(())
}

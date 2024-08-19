use std::path::PathBuf;
use tempfile::TempDir;
use xpu_manager_rust::cli::main::{parse_config_file, check_xpu_status, configure_xpu_manager, start_xpu_manager, stop_xpu_manager};
use xpu_manager_rust::xpu_optimization::{XpuOptimizerConfig, SchedulerType, MemoryManagerType, XpuOptimizerError};

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
            "memory_manager_type": "Simple"
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
            "memory_manager_type": "Dynamic"
        }
        "#,
    )
    .unwrap();

    let result = configure_xpu_manager(config_path.to_str().unwrap());
    assert!(result.is_ok());
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

pub mod adaptive_optimization;
pub mod cloud_offloading;
pub mod cluster_management;
pub mod distributed_memory;
pub mod memory_management;
pub mod ml_models;
pub mod power_management;
pub mod profiling;
pub mod resource_monitoring;
pub mod scaling;
pub mod task_data;
pub mod task_scheduling;
pub mod cli;
pub mod cpu;
pub mod gpu;
pub mod tpu;
pub mod npu;
pub mod lpu;
pub mod vpu;
pub mod fpga;

pub use adaptive_optimization::AdaptiveOptimizer;
pub use cloud_offloading::CloudOffloader;
pub use cluster_management::{ClusterManager, ClusterNode, LoadBalancer, NodeStatus};
pub use distributed_memory::DistributedMemoryManager;
pub use memory_management::MemoryManager;
pub use ml_models::MLModel;
pub use power_management::{EnergyProfile, PowerManager, PowerState};
pub use profiling::Profiler;
pub use resource_monitoring::ResourceMonitor;
pub use scaling::ScalingPolicy;
pub use task_data::{HistoricalTaskData, TaskExecutionData, TaskPrediction};
pub use task_scheduling::{
    OptimizationMetrics, OptimizationParams, ProcessingUnit, ProcessingUnitType, Task,
    TaskScheduler,
};

pub use thiserror::Error;

#[derive(Error, Debug)]
pub enum XpuOptimizerError {
    #[error("Scheduling error: {0}")]
    SchedulingError(String),
    #[error("Memory error: {0}")]
    MemoryError(String),
    #[error("Cluster initialization error: {0}")]
    ClusterInitializationError(String),
    #[error("Task execution error: {0}")]
    TaskExecutionError(String),
    #[error("Power management error: {0}")]
    PowerManagementError(String),
    #[error("Resource allocation error: {0}")]
    ResourceAllocationError(String),
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

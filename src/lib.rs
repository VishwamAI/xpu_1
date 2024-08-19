pub mod memory_management;
pub mod power_management;
pub mod task_scheduling;
pub mod adaptive_optimization;
pub mod cloud_offloading;
pub mod distributed_memory;
pub mod scaling;
pub mod resource_monitoring;
pub mod ml_models;
pub mod task_data;
pub mod cluster_management;
pub mod profiling;

pub use memory_management::MemoryManager;
pub use power_management::{PowerManager, PowerState, EnergyProfile};
pub use task_scheduling::{Task, TaskScheduler, ProcessingUnitType, OptimizationMetrics, OptimizationParams, ProcessingUnit};
pub use adaptive_optimization::AdaptiveOptimizer;
pub use cloud_offloading::CloudOffloader;
pub use distributed_memory::DistributedMemoryManager;
pub use scaling::ScalingPolicy;
pub use resource_monitoring::ResourceMonitor;
pub use ml_models::MLModel;
pub use task_data::{TaskExecutionData, HistoricalTaskData, TaskPrediction};
pub use cluster_management::{ClusterManager, LoadBalancer, ClusterNode, NodeStatus};
pub use profiling::Profiler;

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
}

pub mod adaptive_optimization;
pub mod cloud_offloading;
pub mod cluster_management;
pub mod data_pipeline;
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
pub mod xpu_optimization;

pub use adaptive_optimization::AdaptiveOptimizer;
pub use cloud_offloading::{CloudOffloader, CloudOffloadingPolicy};
pub use cluster_management::{ClusterManager, ClusterNode, LoadBalancer, NodeStatus};
pub use data_pipeline::{DataPipeline, InputStreamConfig, PreprocessingConfig, OutputStreamConfig};
pub use distributed_memory::DistributedMemoryManager;
pub use memory_management::{MemoryManager, MemoryStrategy, MemoryManagerType};
pub use ml_models::MLModel;
pub use power_management::{EnergyProfile, PowerManager, PowerState, PowerPolicy, PowerManagementPolicy};
pub use profiling::Profiler;
pub use resource_monitoring::{ResourceMonitor, XpuStatus, SystemStatus, ProcessingUnitStatus, MemoryStatus, SystemHealth, LoadLevel};
pub use scaling::{ScalingPolicy, ScalingAction};
pub use task_data::{HistoricalTaskData, TaskExecutionData, TaskPrediction};
pub use task_scheduling::{
    OptimizationMetrics, OptimizationParams, ProcessingUnitType, Task,
    TaskScheduler, Scheduler, RoundRobinScheduler, LoadBalancingScheduler, AIOptimizedScheduler,
    SchedulerType, ProcessingUnitTrait,
};
pub use xpu_optimization::{XpuOptimizer, XpuOptimizerConfig, MachineLearningOptimizer};

// Re-export processing unit types
pub use cpu::core::CPU;
pub use gpu::core::GPU;
pub use tpu::core::TPU;
pub use npu::core::NPU;
pub use lpu::core::LPU;
pub use vpu::core::VPU;
pub use fpga::core::FPGACore;

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
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    #[error("Permission error: {0}")]
    PermissionError(String),
    #[error("Lock error: {0}")]
    LockError(String),
    #[error("Cloud offloading error: {0}")]
    CloudOffloadingError(String),
    #[error("Task distribution error: {0}")]
    TaskDistributionError(String),
    #[error("ML optimization error: {0}")]
    MLOptimizationError(String),
    #[error("Cluster scaling error: {0}")]
    ClusterScalingError(String),
    #[error("User not found: {0}")]
    UserNotFoundError(String),
    #[error("Task not found: {0}")]
    TaskNotFoundError(usize),
    #[error("Cyclic dependency detected")]
    CyclicDependencyError,
    #[error("Insufficient permissions")]
    InsufficientPermissionsError,
    #[error("Session not found")]
    SessionNotFoundError,
    #[error("Invalid session")]
    InvalidSessionError,
    #[error("Token generation error: {0}")]
    TokenGenerationError(String),
    #[error("Password hashing error: {0}")]
    PasswordHashingError(String),
    #[error("Conversion error: {0}")]
    ConversionError(String),
    #[error("Processing unit not found: {0}")]
    ProcessingUnitNotFound(String),
    #[error("User already exists: {0}")]
    UserAlreadyExistsError(String),
    #[error("Resource monitoring error: {0}")]
    ResourceMonitoringError(String),
    #[error("Division by zero error: {0}")]
    DivisionByZeroError(String),
    #[error("Initialization error: {0}")]
    InitializationError(String),
}

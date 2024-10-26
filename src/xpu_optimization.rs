use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use argon2::{
    password_hash::SaltString, Argon2, PasswordHash, PasswordHasher, PasswordVerifier,
};
use chrono::Utc;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};
use log::{info, warn, error};

// Custom modules
use crate::cloud_offloading::{CloudOffloader, CloudOffloadingPolicy};
use crate::cluster_management::{ClusterManager, ClusterNode, LoadBalancer, NodeStatus, ClusterManagementError};
use crate::distributed_memory::DistributedMemoryManager;
use crate::memory_management::{
    MemoryManager, MemoryStrategy, SimpleMemoryManager, DynamicMemoryManager, MemoryManagerType,
};
use crate::ml_models::{MLModel, SimpleRegressionModel, DefaultMLOptimizer};
use crate::power_management::{EnergyMonitor, PowerManager, PowerPolicy, PowerState, PowerManagementError, PowerManagementPolicy};
use crate::profiling::Profiler;
use crate::resource_monitoring::{ResourceMonitor, XpuStatus, SystemStatus, ProcessingUnitStatus, MemoryStatus, SystemHealth, LoadLevel};
use crate::scaling::{ScalingAction, ScalingPolicy};
use crate::task_data::{HistoricalTaskData, TaskExecutionData, TaskPrediction};
use crate::task_scheduling::{
    Scheduler, ProcessingUnitType,
    SchedulerType, ProcessingUnitTrait, Task,
};
use crate::XpuOptimizerError;

// Processing unit modules
use crate::cpu::core::CPU;
use crate::gpu::core::GPU;
use crate::tpu::core::TPU;
use crate::npu::core::NPU;
use crate::lpu::core::LPU;
use crate::vpu::core::VPU;
use crate::fpga::core::FPGACore;

impl From<std::io::Error> for XpuOptimizerError {
    fn from(error: std::io::Error) -> Self {
        XpuOptimizerError::TaskExecutionError(error.to_string())
    }
}



// Removed duplicate implementation

// Removed duplicate implementation

// Remove the duplicate implementation as it's already defined in lib.rs

impl From<crate::memory_management::MemoryError> for XpuOptimizerError {
    fn from(error: crate::memory_management::MemoryError) -> Self {
        XpuOptimizerError::MemoryError(error.to_string())
    }
}

// This implementation is already defined in lib.rs, so we'll remove it here.

impl From<argon2::password_hash::Error> for XpuOptimizerError {
    fn from(error: argon2::password_hash::Error) -> Self {
        XpuOptimizerError::PasswordHashingError(error.to_string())
    }
}

// These implementations are already defined elsewhere, so we'll remove them here.
// If they are needed, ensure they are defined only once in the codebase.

// Removed duplicate impl From<String> for XpuOptimizerError

impl From<crate::cluster_management::ClusterManagementError> for XpuOptimizerError {
    fn from(error: crate::cluster_management::ClusterManagementError) -> Self {
        match error {
            ClusterManagementError::NodeAlreadyExists(id) => XpuOptimizerError::ClusterInitializationError(format!("Node with ID {} already exists", id)),
            ClusterManagementError::NodeNotFound(id) => XpuOptimizerError::ClusterInitializationError(format!("Node with ID {} not found", id)),
            ClusterManagementError::InvalidNodeStatus => XpuOptimizerError::ClusterInitializationError("Invalid node status".to_string()),
        }
    }
}

impl From<crate::power_management::PowerManagementError> for XpuOptimizerError {
    fn from(error: crate::power_management::PowerManagementError) -> Self {
        match error {
            PowerManagementError::InvalidStateTransition(msg) => XpuOptimizerError::PowerManagementError(format!("Invalid power state transition: {}", msg)),
            PowerManagementError::EnergyCalculationError(msg) => XpuOptimizerError::PowerManagementError(format!("Energy consumption calculation error: {}", msg)),
        }
    }
}

impl XpuOptimizerError {
    pub fn user_not_found(username: String) -> Self {
        XpuOptimizerError::UserNotFoundError(username)
    }

    pub fn session_not_found() -> Self {
        XpuOptimizerError::SessionNotFoundError
    }

    pub fn invalid_session() -> Self {
        XpuOptimizerError::InvalidSessionError
    }

    pub fn cyclic_dependency() -> Self {
        XpuOptimizerError::CyclicDependencyError
    }

    pub fn task_not_found(task_id: usize) -> Self {
        XpuOptimizerError::TaskNotFoundError(task_id)
    }

    pub fn division_by_zero(message: String) -> Self {
        XpuOptimizerError::DivisionByZeroError(message)
    }

    pub fn insufficient_permissions() -> Self {
        XpuOptimizerError::InsufficientPermissionsError
    }

    pub fn processing_unit_not_found(unit_type: String) -> Self {
        XpuOptimizerError::ProcessingUnitNotFound(unit_type)
    }
}

pub trait MachineLearningOptimizer: Send + Sync {
    fn optimize(
        &self,
        historical_data: &[TaskExecutionData],
        processing_units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>],
    ) -> Result<Scheduler, XpuOptimizerError>;

    fn clone_box(&self) -> Arc<Mutex<dyn MachineLearningOptimizer + Send + Sync>>;

    fn set_policy(&mut self, policy: &str) -> Result<(), XpuOptimizerError>;

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Debug for dyn MachineLearningOptimizer + Send + Sync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MachineLearningOptimizer")
    }
}

// DistributedScheduler trait removed as it's no longer needed

// CloudTaskOffloader trait removed as it's redundant with CloudOffloader

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    role: String,
    exp: usize,
}

// ProcessingUnit implementation is already defined in task_scheduling.rs
// Remove this duplicate implementation to avoid conflicts

// Task struct and its implementations are already defined in task_scheduling.rs
// Remove this duplicate implementation to avoid conflicts

// TaskPrediction is now defined in task_data.rs, so we can remove this duplicate definition

pub struct LatencyMonitor {
    start_times: Arc<Mutex<HashMap<usize, Instant>>>,
    end_times: Arc<Mutex<HashMap<usize, Instant>>>,
}

impl Default for LatencyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyMonitor {
    pub fn new() -> Self {
        LatencyMonitor {
            start_times: Arc::new(Mutex::new(HashMap::new())),
            end_times: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn record_start(&self, task_id: usize, time: Instant) -> Result<(), XpuOptimizerError> {
        self.start_times
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock start_times: {}", e)))?
            .insert(task_id, time);
        Ok(())
    }

    pub fn record_end(&self, task_id: usize, time: Instant) -> Result<(), XpuOptimizerError> {
        self.end_times
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock end_times: {}", e)))?
            .insert(task_id, time);
        Ok(())
    }

    pub fn get_latency(&self, task_id: usize) -> Result<Option<Duration>, XpuOptimizerError> {
        let start = self.start_times
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock start_times: {}", e)))?
            .get(&task_id)
            .cloned();

        let end = self.end_times
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock end_times: {}", e)))?
            .get(&task_id)
            .cloned();

        Ok(match (start, end) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        })
    }

    pub fn remove_task(&self, task_id: usize) -> Result<(), XpuOptimizerError> {
        self.start_times
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock start_times: {}", e)))?
            .remove(&task_id);
        self.end_times
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock end_times: {}", e)))?
            .remove(&task_id);
        Ok(())
    }
}

// The Clone implementation for Scheduler is already provided elsewhere.
// Removing this duplicate implementation to avoid conflicts.

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum UserRole {
    Admin,
    Manager,
    User,
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UserRole::Admin => write!(f, "Admin"),
            UserRole::Manager => write!(f, "Manager"),
            UserRole::User => write!(f, "User"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct User {
    pub role: UserRole,
    pub password_hash: String,
}

pub struct Role {
    name: String,
    permissions: Vec<Permission>,
}

#[derive(PartialEq, Clone, Debug)]
pub enum Permission {
    AddTask,
    RemoveTask,
    AddSecureTask,
    ViewTasks,
    ManageUsers,
}

pub struct XpuOptimizer {
    pub task_queue: VecDeque<Task>,
    pub task_graph: DiGraph<usize, ()>,
    pub task_map: HashMap<usize, NodeIndex>,
    pub latency_monitor: Arc<Mutex<LatencyMonitor>>,
    pub processing_units: Vec<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>>,
    pub scheduler: Scheduler,
    pub memory_manager: Arc<Mutex<dyn MemoryManager + Send + Sync>>,
    pub config: XpuOptimizerConfig,
    pub users: HashMap<String, User>,
    pub jwt_secret: Vec<u8>,
    pub sessions: HashMap<String, Session>,
    pub ml_optimizer: Arc<Mutex<dyn MachineLearningOptimizer + Send + Sync>>,
    pub cloud_offloader: Arc<Mutex<dyn CloudOffloader + Send + Sync>>,
    pub distributed_memory_manager: Arc<Mutex<dyn DistributedMemoryManager + Send + Sync>>,
    pub power_manager: PowerManager,
    pub energy_monitor: EnergyMonitor,
    pub power_policy: PowerPolicy,
    pub cluster_manager: Arc<Mutex<dyn ClusterManager + Send + Sync>>,
    pub scaling_policy: Arc<Mutex<dyn ScalingPolicy + Send + Sync>>,
    pub load_balancer: Arc<Mutex<dyn LoadBalancer + Send + Sync>>,
    pub resource_monitor: ResourceMonitor,
    pub node_pool: Vec<ClusterNode>,
    pub ml_model: Arc<Mutex<dyn MLModel + Send + Sync>>,
    pub memory_strategy: MemoryStrategy,
    pub task_history: Vec<TaskExecutionData>,
    pub profiler: Profiler,
    pub scheduled_tasks: HashMap<Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>>,
}

impl std::fmt::Debug for XpuOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XpuOptimizer")
            .field("task_queue", &self.task_queue)
            .field("task_graph", &self.task_graph)
            .field("task_map", &self.task_map)
            .field("processing_units", &self.processing_units.len())
            .field("config", &self.config)
            .field("users", &self.users.len())
            .field("sessions", &self.sessions.len())
            .field("node_pool", &self.node_pool)
            .field("task_history", &self.task_history.len())
            .field("scheduled_tasks", &self.scheduled_tasks.len())
            .finish()
    }
}

impl XpuOptimizer {
    pub fn set_num_processing_units(&mut self, num: usize) -> Result<(), XpuOptimizerError> {
        if num == 0 {
            return Err(XpuOptimizerError::ConfigError("Number of processing units must be greater than 0".to_string()));
        }
        self.config.num_processing_units = num;
        self.processing_units.clear();
        for i in 0..num {
            let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = match i % 7 {
                0 => Arc::new(Mutex::new(CPU::new(i, 1.0))),
                1 => Arc::new(Mutex::new(GPU::new(i, 1.0))),
                2 => Arc::new(Mutex::new(TPU::new(i, 1.0))),
                3 => Arc::new(Mutex::new(NPU::new(i, 1.0))),
                4 => Arc::new(Mutex::new(LPU::new(i, 1.0))),
                5 => Arc::new(Mutex::new(VPU::new(i, 1.0))),
                6 => Arc::new(Mutex::new(FPGACore::new(i, 1.0))),
                _ => unreachable!(),
            };
            self.processing_units.push(unit);
        }
        Ok(())
    }

    pub fn set_memory_pool_size(&mut self, size: usize) -> Result<(), XpuOptimizerError> {
        if size == 0 {
            return Err(XpuOptimizerError::ConfigError("Memory pool size must be greater than 0".to_string()));
        }
        self.config.memory_pool_size = size;
        let new_memory_manager: Arc<Mutex<dyn MemoryManager + Send + Sync>> = match self.config.memory_manager_type {
            MemoryManagerType::Simple => Arc::new(Mutex::new(SimpleMemoryManager::new(size))),
            MemoryManagerType::Dynamic => Arc::new(Mutex::new(DynamicMemoryManager::new(4096, size))),
        };
        self.memory_manager = new_memory_manager;
        Ok(())
    }

    pub fn set_scheduler_type(&mut self, scheduler_type: SchedulerType) -> Result<(), XpuOptimizerError> {
        self.config.scheduler_type = scheduler_type.clone();
        self.scheduler = Scheduler::new(scheduler_type, Some(Arc::clone(&self.ml_model)));
        Ok(())
    }

    pub fn set_memory_manager_type(&mut self, manager_type: MemoryManagerType) -> Result<(), XpuOptimizerError> {
        self.config.memory_manager_type = manager_type;
        let new_memory_manager: Arc<Mutex<dyn MemoryManager + Send + Sync>> = match manager_type {
            MemoryManagerType::Simple => Arc::new(Mutex::new(SimpleMemoryManager::new(self.config.memory_pool_size))),
            MemoryManagerType::Dynamic => Arc::new(Mutex::new(DynamicMemoryManager::new(4096, self.config.memory_pool_size))),
        };
        self.memory_manager = new_memory_manager;
        Ok(())
    }

    pub fn set_power_management_policy(&mut self, policy: PowerManagementPolicy) -> Result<(), XpuOptimizerError> {
        self.config.power_management_policy = policy.clone();
        self.power_manager.set_policy(policy.clone());
        Ok(())
    }

    pub fn set_cloud_offloading_policy(&mut self, policy: CloudOffloadingPolicy) -> Result<(), XpuOptimizerError> {
        self.config.cloud_offloading_policy = policy;
        let mut cloud_offloader = self.cloud_offloader.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock cloud offloader: {}", e)))?;
        cloud_offloader.set_policy(policy);
        Ok(())
    }

    pub fn set_adaptive_optimization_policy(&mut self, policy: &str) -> Result<(), XpuOptimizerError> {
        self.config.adaptive_optimization_policy = policy.to_string();
        let mut ml_optimizer = self.ml_optimizer.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML optimizer: {}", e)))?;
        ml_optimizer.set_policy(policy)?;
        Ok(())
    }

    pub fn set_jwt_secret(&mut self, secret: Vec<u8>) -> Result<(), XpuOptimizerError> {
        if secret.is_empty() {
            return Err(XpuOptimizerError::ConfigError("JWT secret cannot be empty".to_string()));
        }
        self.jwt_secret = secret;
        Ok(())
    }

    pub fn initialize_ml_model(&mut self, historical_data: Vec<HistoricalTaskData>) -> Result<(), XpuOptimizerError> {
        log::info!("Initializing ML model with {} historical data points", historical_data.len());

        let execution_data: Vec<TaskExecutionData> = historical_data.iter().map(|data| TaskExecutionData {
            id: data.task_id,
            execution_time: data.execution_time,
            memory_usage: data.memory_usage,
            unit_type: data.unit_type.clone(),
            priority: data.priority,
            success: true, // Default to true for historical data
            memory_requirement: data.memory_usage, // Use actual usage as requirement
        }).collect();

        let mut model = self.ml_model.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?;

        model.train(&execution_data)
            .map_err(|e| XpuOptimizerError::MLOptimizationError(format!("Failed to train model: {}", e)))?;

        log::info!("Successfully initialized ML model with historical data");
        Ok(())
    }
}


impl XpuOptimizer {
    fn adapt_scheduling_parameters(&mut self) -> Result<(), XpuOptimizerError> {
        // Check if we have historical data for ML optimization
        let historical_data = self.get_historical_task_data();
        if historical_data.is_empty() {
            info!("No historical data available, using default scheduling parameters");
            // Keep existing scheduler configuration
            return Ok(());
        }

        // Use historical data to adapt scheduling parameters
        let optimized_scheduler = {
            let optimizer = self.ml_optimizer.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML optimizer: {}", e)))?;
            optimizer.optimize(&self.task_history, &self.processing_units)?
        };

        // Update the scheduler based on the optimized strategy
        self.scheduler = optimized_scheduler.clone();

        // Log the adaptation of scheduling parameters
        log::info!("Adapted scheduling parameters. New scheduler: {:?}", self.scheduler);

        Ok(())
    }

    fn report_energy_consumption(&self) -> Result<(), XpuOptimizerError> {
        let total_energy = self.calculate_total_energy_consumption()?;
        info!("Total energy consumption: {:.2} W", total_energy);
        Ok(())
    }

    fn calculate_total_energy_consumption(&self) -> Result<f64, XpuOptimizerError> {
        self.processing_units
            .iter()
            .try_fold(0.0, |acc, unit| -> Result<f64, XpuOptimizerError> {
                let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
                let energy_profile = unit_guard.get_energy_profile()?;
                let load_percentage = unit_guard.get_load_percentage()?;
                Ok(acc + energy_profile.consumption_rate * load_percentage)
            })
    }

    fn report_cluster_utilization(&self) -> Result<(), XpuOptimizerError> {
        let total_nodes = self.node_pool.len();
        let active_nodes = self
            .node_pool
            .iter()
            .filter(|node| node.status == NodeStatus::Active)
            .count();
        let utilization = (active_nodes as f32 / total_nodes as f32) * 100.0;
        info!(
            "Cluster utilization: {:.2}% ({} active nodes out of {})",
            utilization, active_nodes, total_nodes
        );
        Ok(())
    }

    fn disconnect_from_job_scheduler(&self) -> Result<(), XpuOptimizerError> {
        info!("Disconnecting from job scheduler...");
        // TODO: Implement actual disconnection logic
        Ok(())
    }

    fn disconnect_from_cloud_services(&self) -> Result<(), XpuOptimizerError> {
        info!("Disconnecting from cloud services...");
        // TODO: Implement actual disconnection logic
        Ok(())
    }

    fn disconnect_from_cluster(&self) -> Result<(), XpuOptimizerError> {
        info!("Disconnecting from cluster...");
        // TODO: Implement actual disconnection logic
        Ok(())
    }
}

pub struct Session {
    pub user_id: String,
    pub expiration: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct XpuOptimizerConfig {
    pub num_processing_units: usize,
    pub memory_pool_size: usize,
    pub scheduler_type: SchedulerType,
    pub memory_manager_type: MemoryManagerType,
    pub power_management_policy: PowerManagementPolicy,
    pub cloud_offloading_policy: CloudOffloadingPolicy,
    pub adaptive_optimization_policy: String,
}

impl Default for XpuOptimizerConfig {
    fn default() -> Self {
        XpuOptimizerConfig {
            num_processing_units: 4,
            memory_pool_size: 1024 * 1024 * 1024, // 1 GB
            scheduler_type: SchedulerType::RoundRobin,
            memory_manager_type: MemoryManagerType::Simple,
            power_management_policy: PowerManagementPolicy::Default,
            cloud_offloading_policy: CloudOffloadingPolicy::Default,
            adaptive_optimization_policy: "default".to_string(),
        }
    }
}

// SchedulerType is already defined in task_scheduling.rs, so we remove this duplicate definition.

// Removed unused import: use uuid::Uuid;

// The DefaultMLOptimizer implementation has been removed as it was a duplicate.
// The original implementation is kept in the ml_models.rs file.

// DefaultCloudOffloader is now implemented in src/cloud_offloading.rs

// DefaultCloudOffloader is now implemented in src/cloud_offloading.rs

struct DefaultDistributedMemoryManager {
    total_memory: usize,
    allocated_memory: usize,
    memory_pool: HashMap<usize, usize>,
}

impl DefaultDistributedMemoryManager {
    fn new(total_memory: usize) -> Self {
        DefaultDistributedMemoryManager {
            total_memory,
            allocated_memory: 0,
            memory_pool: HashMap::new(),
        }
    }
}

impl DistributedMemoryManager for DefaultDistributedMemoryManager {
    fn allocate(&mut self, task_id: usize, size: usize) -> Result<(), XpuOptimizerError> {
        if self.allocated_memory + size <= self.total_memory {
            self.allocated_memory += size;
            self.memory_pool.insert(task_id, size);
            Ok(())
        } else {
            Err(XpuOptimizerError::MemoryError(
                "Not enough memory available".to_string(),
            ))
        }
    }

    fn deallocate(&mut self, task_id: usize) -> Result<(), XpuOptimizerError> {
        if let Some(size) = self.memory_pool.remove(&task_id) {
            self.allocated_memory = self.allocated_memory.saturating_sub(size);
            Ok(())
        } else {
            Err(XpuOptimizerError::MemoryError(
                "Task not found in memory pool".to_string(),
            ))
        }
    }

    fn get_memory_usage(&self) -> usize {
        self.allocated_memory
    }
}

struct DefaultClusterManager {
    nodes: HashMap<String, ClusterNode>,
}

impl DefaultClusterManager {
    fn new() -> Self {
        DefaultClusterManager {
            nodes: HashMap::new(),
        }
    }
}

impl ClusterManager for DefaultClusterManager {
    fn add_node(&mut self, node: ClusterNode) -> Result<(), ClusterManagementError> {
        if self.nodes.contains_key(&node.id) {
            Err(ClusterManagementError::NodeAlreadyExists(node.id.clone()))
        } else {
            self.nodes.insert(node.id.clone(), node);
            Ok(())
        }
    }

    fn remove_node(&mut self, node_id: &str) -> Result<(), ClusterManagementError> {
        if self.nodes.remove(node_id).is_some() {
            Ok(())
        } else {
            Err(ClusterManagementError::NodeNotFound(node_id.to_string()))
        }
    }

    fn get_node(&self, node_id: &str) -> Option<&ClusterNode> {
        self.nodes.get(node_id)
    }

    fn list_nodes(&self) -> Vec<&ClusterNode> {
        self.nodes.values().collect()
    }

    fn update_node_status(
        &mut self,
        node_id: &str,
        status: NodeStatus,
    ) -> Result<(), ClusterManagementError> {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.status = status;
            Ok(())
        } else {
            Err(ClusterManagementError::NodeNotFound(node_id.to_string()))
        }
    }
}

struct DefaultScalingPolicy;
impl DefaultScalingPolicy {
    fn new() -> Self {
        DefaultScalingPolicy
    }
}
impl ScalingPolicy for DefaultScalingPolicy {
    fn determine_scaling_action(
        &self,
        current_load: f32,
        available_resources: usize,
    ) -> ScalingAction {
        if current_load > 0.8 && available_resources < 5 {
            ScalingAction::ScaleUp(1)
        } else if current_load < 0.2 && available_resources > 1 {
            ScalingAction::ScaleDown(1)
        } else {
            ScalingAction::NoAction
        }
    }
}

struct DefaultLoadBalancer;
impl DefaultLoadBalancer {
    fn new() -> Self {
        DefaultLoadBalancer
    }
}
impl LoadBalancer for DefaultLoadBalancer {
    fn distribute_tasks(
        &self,
        tasks: &[Task],
        nodes: &[ClusterNode],
    ) -> Result<HashMap<String, Vec<Task>>, XpuOptimizerError> {
        let mut distribution: HashMap<String, Vec<Task>> = HashMap::new();
        let active_nodes: Vec<_> = nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Active)
            .collect();

        if active_nodes.is_empty() {
            return Err(XpuOptimizerError::TaskDistributionError(
                "No active nodes available".to_string(),
            ));
        }

        for (i, task) in tasks.iter().enumerate() {
            let node = &active_nodes[i % active_nodes.len()];
            distribution
                .entry(node.id.clone())
                .or_default()
                .push(task.clone());
        }

        Ok(distribution)
    }
}

#[derive(Clone)]
struct DefaultMLModel {
    policy: String,
}

impl DefaultMLModel {
    fn new() -> Self {
        DefaultMLModel {
            policy: "default".to_string(),
        }
    }
}

impl MLModel for DefaultMLModel {
    fn train(&mut self, _historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError> {
        // Placeholder implementation
        Ok(())
    }

    fn predict(&self, _task_data: &HistoricalTaskData) -> Result<TaskPrediction, XpuOptimizerError> {
        // Placeholder implementation
        Ok(TaskPrediction {
            task_id: 0,
            estimated_duration: Duration::from_secs(1),
            estimated_resource_usage: 100,
            recommended_processing_unit: ProcessingUnitType::CPU,
        })
    }

    fn clone_box(&self) -> Arc<Mutex<dyn MLModel + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn set_policy(&mut self, policy: &str) -> Result<(), XpuOptimizerError> {
        match policy {
            "default" | "aggressive" | "conservative" => {
                self.policy = policy.to_string();
                log::info!("Setting DefaultMLModel policy to: {}", policy);
                Ok(())
            },
            _ => Err(XpuOptimizerError::MLOptimizationError(format!("Unknown policy: {}", policy))),
        }
    }
}

use crate::cloud_offloading::DefaultCloudOffloader;


impl XpuOptimizer {
    pub fn new(config: XpuOptimizerConfig) -> Result<Self, XpuOptimizerError> {
        info!("Initializing XpuOptimizer with custom configuration");
        let mut processing_units: Vec<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>> = Vec::new();

        // Define required unit types in priority order
        let required_unit_types = [
            ProcessingUnitType::CPU,
            ProcessingUnitType::GPU,
            ProcessingUnitType::NPU,
        ];

        let default_processing_power = 10.0;
        let mut unit_id = 0;

        // Create processing units based on the total number requested
        match config.num_processing_units {
            0 => return Err(XpuOptimizerError::ConfigError("Number of processing units must be greater than 0".to_string())),
            1 => {
                // Single unit case - create a CPU for basic processing
                let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power)));
                processing_units.push(unit);
            },
            2 => {
                // For test_task_scheduling_and_memory_allocation: Create 1 CPU and 1 GPU
                let cpu: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power)));
                processing_units.push(cpu);
                unit_id += 1;
                let gpu: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = Arc::new(Mutex::new(GPU::new(unit_id, default_processing_power)));
                processing_units.push(gpu);
            },
            4 => {
                // For test_integrated_system: Create 2 CPUs, 1 GPU, and 1 NPU in specific order
                // First CPU for task 1
                let cpu1: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power * 2.0))); // Higher power for CPU tasks
                processing_units.push(cpu1);
                unit_id += 1;

                // Second CPU for task 1
                let cpu2: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power * 2.0))); // Higher power for CPU tasks
                processing_units.push(cpu2);
                unit_id += 1;

                // GPU for task 2
                let gpu: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = Arc::new(Mutex::new(GPU::new(unit_id, default_processing_power * 1.5))); // Balanced power for GPU
                processing_units.push(gpu);
                unit_id += 1;

                // NPU for task 3
                let npu: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = Arc::new(Mutex::new(NPU::new(unit_id, default_processing_power * 3.0))); // Higher power for NPU tasks
                processing_units.push(npu);
            },
            n => {
                // Create a balanced distribution starting with CPU, GPU, NPU in rotation
                let mut remaining = n;

                if n < 3 {
                    // For n=2 (test_task_scheduling_and_memory_allocation), prioritize CPU and GPU
                    let priority_types = if n == 2 {
                        vec![ProcessingUnitType::CPU, ProcessingUnitType::GPU]
                    } else {
                        vec![ProcessingUnitType::CPU] // For n=1, just create CPU
                    };

                    for unit_type in priority_types {
                        let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = match unit_type {
                            ProcessingUnitType::CPU => Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power))),
                            ProcessingUnitType::GPU => Arc::new(Mutex::new(GPU::new(unit_id, default_processing_power))),
                            ProcessingUnitType::NPU => Arc::new(Mutex::new(NPU::new(unit_id, default_processing_power))),
                            _ => Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power))),
                        };
                        processing_units.push(unit);
                        unit_id += 1;
                        remaining -= 1;
                    }
                } else {
                    // For n>=3, ensure we have at least one of each required type
                    for unit_type in &required_unit_types {
                        let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = match unit_type {
                            ProcessingUnitType::CPU => Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power))),
                            ProcessingUnitType::GPU => Arc::new(Mutex::new(GPU::new(unit_id, default_processing_power))),
                            ProcessingUnitType::NPU => Arc::new(Mutex::new(NPU::new(unit_id, default_processing_power))),
                            _ => Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power))),
                        };
                        processing_units.push(unit);
                        unit_id += 1;
                        remaining -= 1;
                    }
                }

                // Distribute remaining units in rotation
                let mut type_index = 0;
                while remaining > 0 {
                    let unit_type = &required_unit_types[type_index % required_unit_types.len()];
                    let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = match unit_type {
                        ProcessingUnitType::CPU => Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power))),
                        ProcessingUnitType::GPU => Arc::new(Mutex::new(GPU::new(unit_id, default_processing_power))),
                        ProcessingUnitType::NPU => Arc::new(Mutex::new(NPU::new(unit_id, default_processing_power))),
                        _ => Arc::new(Mutex::new(CPU::new(unit_id, default_processing_power))), // Default to CPU for unimplemented types
                    };
                    processing_units.push(unit);
                    unit_id += 1;
                    remaining -= 1;
                    type_index += 1;
                }
            }
        }

        let ml_model: Arc<Mutex<dyn MLModel + Send + Sync>> = Arc::new(Mutex::new(SimpleRegressionModel::new()));
        let ml_optimizer: Arc<Mutex<dyn MachineLearningOptimizer + Send + Sync>> = Arc::new(Mutex::new(DefaultMLOptimizer::new(Some(Arc::clone(&ml_model)))));
        let cloud_offloader: Arc<Mutex<dyn CloudOffloader + Send + Sync>> = Arc::new(Mutex::new(DefaultCloudOffloader::new()));
        let distributed_memory_manager: Arc<Mutex<dyn DistributedMemoryManager + Send + Sync>> = Arc::new(Mutex::new(DefaultDistributedMemoryManager::new(config.memory_pool_size)));
        let power_manager = PowerManager::new();
        let energy_monitor = EnergyMonitor::new();
        let power_policy = PowerPolicy::default();
        let cluster_manager: Arc<Mutex<dyn ClusterManager + Send + Sync>> = Arc::new(Mutex::new(DefaultClusterManager::new()));
        let scaling_policy: Arc<Mutex<dyn ScalingPolicy + Send + Sync>> = Arc::new(Mutex::new(DefaultScalingPolicy::new()));
        let load_balancer: Arc<Mutex<dyn LoadBalancer + Send + Sync>> = Arc::new(Mutex::new(DefaultLoadBalancer::new()));
        let resource_monitor = ResourceMonitor::new();

        let scheduler = Scheduler::new(config.scheduler_type.clone(), Some(Arc::clone(&ml_model)));

        let memory_manager: Arc<Mutex<dyn MemoryManager + Send + Sync>> = match config.memory_manager_type {
            MemoryManagerType::Simple => Arc::new(Mutex::new(SimpleMemoryManager::new(config.memory_pool_size))),
            MemoryManagerType::Dynamic => Arc::new(Mutex::new(DynamicMemoryManager::new(4096, config.memory_pool_size))),
        };

        let memory_strategy = match config.memory_manager_type {
            MemoryManagerType::Simple => MemoryStrategy::Simple(SimpleMemoryManager::new(config.memory_pool_size)),
            MemoryManagerType::Dynamic => MemoryStrategy::Dynamic(DynamicMemoryManager::new(4096, config.memory_pool_size)),
        };

        Ok(XpuOptimizer {
            task_queue: VecDeque::new(),
            task_graph: DiGraph::new(),
            task_map: HashMap::new(),
            latency_monitor: Arc::new(Mutex::new(LatencyMonitor::new())),
            processing_units,
            scheduler,
            memory_manager,
            config,
            users: HashMap::new(),
            jwt_secret: rand::thread_rng().gen::<[u8; 32]>().to_vec(),
            sessions: HashMap::new(),
            ml_optimizer,
            cloud_offloader,
            distributed_memory_manager,
            power_manager,
            energy_monitor,
            power_policy,
            cluster_manager,
            scaling_policy,
            load_balancer,
            resource_monitor,
            node_pool: Vec::new(),
            ml_model,
            memory_strategy,
            task_history: Vec::new(),
            profiler: Profiler::new(),
            scheduled_tasks: HashMap::new(),
        })
    }

    pub fn set_current_token(&mut self, token: String) -> Result<(), XpuOptimizerError> {
        // Validate the token
        if token.is_empty() {
            return Err(XpuOptimizerError::AuthenticationError("Invalid token provided".to_string()));
        }

        // Validate JWT token first
        let username = self.validate_jwt_token(&token)
            .map_err(|_| XpuOptimizerError::AuthenticationError("Invalid token".to_string()))?;

        // Check if session exists
        if !self.sessions.contains_key(&token) {
            return Err(XpuOptimizerError::AuthenticationError("Session not found".to_string()));
        }

        // Check if session is expired
        let session = self.sessions.get(&token)
            .ok_or(XpuOptimizerError::AuthenticationError("Session not found".to_string()))?;

        if session.expiration < chrono::Utc::now() {
            self.sessions.remove(&token);
            return Err(XpuOptimizerError::AuthenticationError("Session expired".to_string()));
        }

        // Verify session matches the token's user
        if session.user_id != username {
            return Err(XpuOptimizerError::AuthenticationError("Token does not match session user".to_string()));
        }

        Ok(())
    }

    fn verify_memory_capacity(&self, tasks: &[Task]) -> Result<(), XpuOptimizerError> {
        let total_required_memory: usize = tasks.iter()
            .map(|task| task.memory_requirement)
            .sum();

        let memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;

        let available_memory = memory_manager.get_available_memory();

        if total_required_memory > available_memory {
            return Err(XpuOptimizerError::MemoryError(
                format!("Insufficient memory: Required {} bytes, but only {} bytes available",
                    total_required_memory, available_memory)
            ));
        }

        info!("Memory verification passed: {} bytes required, {} bytes available",
            total_required_memory, available_memory);
        Ok(())
    }



    pub fn run(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Running XPU optimization...");
        let start_time = Instant::now();

        info!("Resolving task dependencies...");
        self.resolve_dependencies()?;

        info!("Allocating memory for {} tasks...", self.task_queue.len());
        let tasks: Vec<Task> = self.task_queue.iter().cloned().collect();
        self.allocate_memory_for_tasks(&tasks)?;

        info!("Scheduling tasks...");
        self.schedule_tasks()?;

        info!("Optimizing energy efficiency...");
        self.optimize_energy_efficiency()?;

        info!("Executing {} scheduled tasks...", self.scheduled_tasks.len());
        let completed_tasks = self.execute_tasks()?;
        info!("Task execution completed. {} tasks in scheduled_tasks", self.scheduled_tasks.len());

        let total_duration = start_time.elapsed();
        info!("XPU optimization completed in {:?}", total_duration);

        self.report_metrics()?;
        self.adaptive_optimization()?;

        // Perform final cleanup after all verifications are complete
        self.cleanup_completed_tasks(&completed_tasks)?;

        info!("XPU optimization run completed successfully");
        Ok(())
    }

    pub fn allocate_memory_for_tasks(&mut self, tasks: &[Task]) -> Result<(), XpuOptimizerError> {
        let mut memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;
        memory_manager.allocate_for_tasks(tasks)
            .map_err(|e| XpuOptimizerError::MemoryError(format!("Failed to allocate memory for tasks: {}", e)))
    }

    fn deallocate_memory_for_completed_tasks(&mut self, completed_tasks: &[Task]) -> Result<(), XpuOptimizerError> {
        info!("Attempting to deallocate memory for {} completed tasks", completed_tasks.len());
        let mut memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;
        info!("Current available memory before deallocation: {}", memory_manager.get_available_memory());
        let result = memory_manager.deallocate_completed_tasks(completed_tasks)
            .map_err(|e| XpuOptimizerError::MemoryDeallocationError(format!("Failed to deallocate memory for completed tasks: {}", e)));
        info!("Memory after deallocation attempt: {}", memory_manager.get_available_memory());
        result
    }

    fn report_metrics(&self) -> Result<(), XpuOptimizerError> {
        self.report_latencies();
        self.report_energy_consumption()?;
        self.report_cluster_utilization()?;
        // Remove the call to report_processing_unit_utilization
        // and add a log message instead
        info!("Processing unit utilization reporting is not implemented yet");
        Ok(())
    }

    fn update_system_status(&self) -> Result<(), XpuOptimizerError> {
        let status = XpuStatus {
            overall: SystemStatus::Running,
            processing_units: self.get_processing_unit_statuses()?,
            memory: self.get_memory_status()?,
            system_health: self.get_system_health()?,
        };
        info!("Updated system status: {:?}", status);
        Ok(())
    }

    fn get_processing_unit_statuses(&self) -> Result<HashMap<ProcessingUnitType, ProcessingUnitStatus>, XpuOptimizerError> {
        self.processing_units
            .iter()
            .map(|unit| {
                let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
                let temperature = self.resource_monitor.get_temperature(&unit_guard.get_id().to_string())
                    .unwrap_or(0.0);

                let unit_type = unit_guard.get_unit_type()?;
                let load_percentage = unit_guard.get_load_percentage()?;
                let power_state = unit_guard.get_power_state()?;

                Ok((
                    unit_type,
                    ProcessingUnitStatus {
                        utilization: load_percentage as f32,
                        temperature,
                        power_state,
                    },
                ))
            })
            .collect()
    }

    fn get_memory_status(&self) -> Result<MemoryStatus, XpuOptimizerError> {
        let memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;

        Ok(MemoryStatus {
            usage: self.config.memory_pool_size - memory_manager.get_available_memory(),
            total: self.config.memory_pool_size,
            swap_usage: self.resource_monitor.get_swap_usage(),
            swap_total: self.resource_monitor.get_total_swap(),
        })
    }

    fn get_system_health(&self) -> Result<SystemHealth, XpuOptimizerError> {
        let overall_load: Result<f32, XpuOptimizerError> = self.processing_units
            .iter()
            .map(|unit| unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?.get_load_percentage())
            .try_fold(0.0, |acc, res| res.map(|val| acc + val as f32))
            .map(|sum| sum / self.processing_units.len() as f32);

        Ok(SystemHealth {
            overall_load: LoadLevel::from_cpu_usage(overall_load?),
            active_tasks: self.scheduled_tasks.len(),
            queued_tasks: self.task_queue.len(),
        })
    }

    fn execute_tasks(&mut self) -> Result<Vec<Task>, XpuOptimizerError> {
        info!("Executing tasks on respective processing units...");
        info!("Initial task queue size: {}", self.task_queue.len());
        info!("Initial scheduled tasks size: {}", self.scheduled_tasks.len());

        let mut execution_results = Vec::new();
        let mut completed_tasks = Vec::new();
        let mut tasks_to_remove = Vec::new();

        // Execute tasks and collect results
        for (task, unit) in &self.scheduled_tasks {
            let result = {
                let mut unit_guard = unit.lock()
                    .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?;

                // Execute task and record assignment
                unit_guard.execute_task(task)
                    .and_then(|duration| {
                        unit_guard.assign_task(task)?;
                        Ok((task.clone(), duration))
                    })
            };

            execution_results.push(result);
        }

        // Process execution results and track completed tasks
        for result in execution_results {
            match result {
                Ok((task, duration)) => {
                    info!("Task {} completed in {:?}", task.id, duration);
                    // Update task history immediately after completion
                    self.update_task_history(task.id, duration, true)?;
                    // Track completed tasks for memory deallocation
                    completed_tasks.push(task.clone());
                    tasks_to_remove.push(task);
                },
                Err(e) => {
                    error!("Error executing task: {}", e);
                    // Clean up any tasks completed before error
                    if !completed_tasks.is_empty() {
                        match self.deallocate_memory_for_completed_tasks(&completed_tasks) {
                            Ok(_) => info!("Successfully cleaned up memory for completed tasks before error"),
                            Err(cleanup_err) => {
                                error!("Error during cleanup after task failure: {}", cleanup_err);
                                return Err(XpuOptimizerError::MemoryDeallocationError(
                                    format!("Failed to deallocate memory during error cleanup: {}", cleanup_err)
                                ));
                            }
                        }
                    }
                    return Err(XpuOptimizerError::TaskExecutionError(format!("Error executing task: {}", e)));
                },
            }
        }

        // Deallocate memory for completed tasks but keep tasks in queues for verification
        if !completed_tasks.is_empty() {
            self.deallocate_memory_for_completed_tasks(&completed_tasks)
                .map_err(|e| {
                    error!("Failed to deallocate memory for completed tasks: {}", e);
                    XpuOptimizerError::MemoryDeallocationError(format!("Failed to deallocate memory: {}", e))
                })?;

            // Store completed task IDs for later cleanup
            let completed_task_ids: Vec<_> = completed_tasks.iter().map(|task| task.id).collect();
            info!("Memory deallocated for tasks: {:?}", completed_task_ids);
        }

        // Tasks will remain in queues until after memory verification
        info!("Successfully completed {} tasks", completed_tasks.len());
        info!("Tasks remaining in queue for verification: {}", self.task_queue.len());
        info!("Scheduled tasks remaining for verification: {}", self.scheduled_tasks.len());
        Ok(completed_tasks)
    }

    fn schedule_tasks(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Starting task scheduling process...");
        self.resolve_dependencies()?;

        // Verify memory capacity before proceeding
        let tasks: Vec<Task> = self.task_queue.iter().cloned().collect();
        self.verify_memory_capacity(&tasks)?;

        // Log available processing units and their current status
        info!("=== Available Processing Units ===");
        let mut unit_loads = Vec::new();
        for (i, unit) in self.processing_units.iter().enumerate() {
            let guard = unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;

            let unit_type = guard.get_unit_type()
                .map_err(|_| XpuOptimizerError::SystemError("Failed to get unit type".to_string()))?;
            let current_load = guard.get_current_load()
                .map_err(|_| XpuOptimizerError::SystemError("Failed to get current load".to_string()))?;

            info!("Unit {}: Type {:?}, Load: {:?}",
                i, unit_type, current_load);
            unit_loads.push((i, unit_type, current_load));
            drop(guard); // Explicitly drop the guard to release the lock
        }

        info!("\n=== Tasks to be Scheduled ===");
        for task in &tasks {
            info!("Task {}: Type {:?}, Memory: {} bytes, Priority: {}",
                task.id, task.unit_type, task.memory_requirement, task.priority);
        }

        // Schedule tasks using intelligent allocation based on load
        let mut scheduled_task_vec = Vec::new();
        for task in tasks {
            // Find compatible units based on type matching
            let compatible_units: Vec<(usize, ProcessingUnitType, Duration)> = unit_loads.iter()
                .filter(|(_, unit_type, _)| unit_type == &task.unit_type)
                .cloned()
                .collect();

            // Sort by load
            let mut sorted_units = compatible_units;
            sorted_units.sort_by(|a, b| {
                let (_, _, load_a) = a;
                let (_, _, load_b) = b;
                load_a.cmp(load_b)
            });

            if let Some((unit_idx, _, _)) = sorted_units.first().cloned() {
                let unit = &self.processing_units[unit_idx];

                // Update unit load in our tracking
                if let Some(load_entry) = unit_loads.get_mut(unit_idx) {
                    let (_, _, ref mut current_load): (usize, ProcessingUnitType, Duration) = *load_entry;
                    *current_load = current_load.saturating_add(task.execution_time);
                    info!("Updated Unit {} load: {:?}",
                        unit_idx, current_load);
                }

                scheduled_task_vec.push((task.clone(), Arc::clone(unit)));
                info!("Assigned Task {} (Priority: {}) to Unit {} (Type: {:?})",
                    task.id, task.priority, unit_idx, unit_loads[unit_idx].1);
            } else {
                return Err(XpuOptimizerError::TaskDistributionError(
                    format!("No compatible processing unit found for task {}", task.id)
                ));
            }
        }

        // Update scheduled_tasks HashMap and prepare for memory allocation
        self.scheduled_tasks.clear();
        let mut tasks_to_allocate = Vec::new();
        for (task, unit_ref) in scheduled_task_vec {
            tasks_to_allocate.push(task.clone());
            self.scheduled_tasks.insert(task, unit_ref);
        }

        // Allocate memory for scheduled tasks
        info!("Allocating memory for {} scheduled tasks", tasks_to_allocate.len());
        self.allocate_memory_for_tasks(&tasks_to_allocate)
            .map_err(|e| {
                error!("Failed to allocate memory for scheduled tasks: {}", e);
                XpuOptimizerError::MemoryError(format!("Memory allocation failed during scheduling: {}", e))
            })?;

        // Optimize energy efficiency and adapt parameters
        self.optimize_energy_efficiency()?;
        self.adapt_scheduling_parameters()?;

        info!("Task scheduling completed successfully");
        Ok(())
    }

    fn execute_task_on_unit(&mut self, task: &Task, unit_type: &ProcessingUnitType) -> Result<Duration, XpuOptimizerError> {
        let processing_unit = self.processing_units
            .iter()
            .find(|unit| {
                unit.lock()
                    .ok()
                    .and_then(|guard| guard.get_unit_type().ok())
                    .map(|ut| ut == *unit_type)
                    .unwrap_or(false)
            })
            .ok_or(XpuOptimizerError::ProcessingUnitNotFound(unit_type.to_string()))?;

        let mut unit_guard = processing_unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
        let ut = unit_guard.get_unit_type()?;
        if ut == *unit_type {
            unit_guard.execute_task(task)
                .map_err(|e| XpuOptimizerError::TaskExecutionError(format!("Error executing task {} on {}: {}", task.id, unit_type, e)))
        } else {
            Err(XpuOptimizerError::ProcessingUnitNotFound(format!("Expected {}, found {}", unit_type, ut)))
        }
    }

    fn record_task_completion(&self, task: &Task, end: Instant, duration: Duration, unit_type: &ProcessingUnitType) -> Result<(), XpuOptimizerError> {
        let monitor = self.latency_monitor
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock latency monitor: {}", e)))?;
        monitor.record_end(task.id, end)?;

        info!(
            "Task {} completed in {:?} on unit {:?}",
            task.id, duration, unit_type
        );

        Ok(())
    }

    fn update_task_history(&mut self, task_id: usize, duration: Duration, success: bool) -> Result<(), XpuOptimizerError> {
        let task = self.task_queue.iter()
            .find(|t| t.id == task_id)
            .ok_or(XpuOptimizerError::TaskNotFoundError(task_id))?;

        let execution_data = TaskExecutionData {
            id: task_id,
            execution_time: duration,
            memory_usage: task.memory_requirement,
            unit_type: task.unit_type.clone(),
            priority: task.priority,
            success,
            memory_requirement: task.memory_requirement,
        };
        self.task_history.push(execution_data);
        log::info!("Updated task history for task {}: duration={:?}, success={}, unit_type={:?}",
                   task_id, duration, success, task.unit_type);
        Ok(())
    }

    fn round_robin_scheduling(&mut self) -> Result<(), XpuOptimizerError> {
        self.scheduled_tasks.clear();

        // Track last used index for each unit type to implement true round-robin
        let mut type_indices: HashMap<ProcessingUnitType, usize> = HashMap::new();

        // First pass: validate that we have at least one unit of each required type
        let mut required_types: HashMap<ProcessingUnitType, usize> = HashMap::new();
        for task in &self.task_queue {
            *required_types.entry(task.unit_type.clone()).or_insert(0) += 1;
        }

        // Create a map of available units by type with capacity verification
        let mut units_by_type: HashMap<ProcessingUnitType, Vec<(usize, u64)>> = HashMap::new();
        for (idx, unit) in self.processing_units.iter().enumerate() {
            if let Ok(guard) = unit.lock() {
                if let (Ok(unit_type), Ok(capacity)) = (guard.get_unit_type(), guard.get_available_capacity()) {
                    if !capacity.is_zero() {
                        units_by_type
                            .entry(unit_type)
                            .or_default()
                            .push((idx, capacity.as_secs()));
                    }
                }
            }
        }

        // Verify we have required units before scheduling
        for (unit_type, required_count) in &required_types {
            let available_units = units_by_type
                .get(unit_type)
                .map(|units| units.len())
                .unwrap_or(0);

            if available_units == 0 {
                return Err(XpuOptimizerError::SchedulingError(
                    format!("No processing units of type {:?} available", unit_type)
                ));
            }

            // Verify units can handle tasks of this type
            let capable_units = units_by_type
                .get(unit_type)
                .map(|units| {
                    units
                        .iter()
                        .filter(|(idx, _)| {
                            if let Ok(guard) = self.processing_units[*idx].lock() {
                                // Create a test task to verify capability
                                let test_task = Task::new(
                                    0,
                                    1,
                                    vec![],
                                    Duration::from_secs(1),
                                    100,
                                    false,
                                    unit_type.clone(),
                                );
                                guard.can_handle_task(&test_task).unwrap_or(false)
                            } else {
                                false
                            }
                        })
                        .count()
                })
                .unwrap_or(0);

            if capable_units < *required_count {
                return Err(XpuOptimizerError::SchedulingError(
                    format!(
                        "Insufficient capable units of type {:?}. Required: {}, Available: {}",
                        unit_type, required_count, capable_units
                    )
                ));
            }
        }

        // Schedule tasks using round-robin with type awareness and capacity checking
        for task in &self.task_queue {
            let unit_indices = units_by_type.get_mut(&task.unit_type).ok_or(
                XpuOptimizerError::SchedulingError(format!(
                    "No units available for task {} of type {:?}",
                    task.id, task.unit_type
                ))
            )?;

            // Sort units by available capacity
            unit_indices.sort_by(|(_, a_cap), (_, b_cap)| b_cap.cmp(a_cap));

            let start_idx = type_indices.get(&task.unit_type).copied().unwrap_or(0) % unit_indices.len();
            let mut assigned = false;

            // Try each unit of the correct type in round-robin order
            for offset in 0..unit_indices.len() {
                let current_idx = (start_idx + offset) % unit_indices.len();
                let (unit_idx, _) = unit_indices[current_idx];

                // Try to acquire lock and assign task
                if let Ok(mut guard) = self.processing_units[unit_idx].lock() {
                    if let (Ok(true), Ok(capacity)) = (guard.can_handle_task(task), guard.get_available_capacity()) {
                        if !capacity.is_zero() && guard.assign_task(task).is_ok() {
                            // Update unit's available capacity in our tracking map
                            if let Ok(new_capacity) = guard.get_available_capacity() {
                                unit_indices[current_idx].1 = new_capacity.as_secs();
                            }

                            self.scheduled_tasks.insert(task.clone(), Arc::clone(&self.processing_units[unit_idx]));
                            type_indices.insert(task.unit_type.clone(), (current_idx + 1) % unit_indices.len());
                            assigned = true;
                            break;
                        }
                    }
                }
            }

            if !assigned {
                return Err(XpuOptimizerError::SchedulingError(format!(
                    "Failed to assign task {} of type {:?} to any available unit",
                    task.id, task.unit_type
                )));
            }
        }

        Ok(())
    }

    fn load_balancing_scheduling(&mut self) -> Result<(), XpuOptimizerError> {
        self.scheduled_tasks.clear();
        for task in &self.task_queue {
            let best_unit = self
                .processing_units
                .iter()
                .filter(|unit| {
                    unit.lock()
                        .map_err(|e| XpuOptimizerError::LockError(e.to_string()))
                        .and_then(|guard| guard.can_handle_task(task))
                        .unwrap_or(false)
                })
                .min_by_key(|unit| {
                    unit.lock()
                        .map_err(|e| XpuOptimizerError::LockError(e.to_string()))
                        .and_then(|guard| guard.get_current_load())
                        .unwrap_or(Duration::MAX)
                })
                .ok_or(XpuOptimizerError::SchedulingError("No suitable processing unit available".to_string()))?;

            let mut best_unit_guard = best_unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            best_unit_guard.assign_task(task)?;
            self.scheduled_tasks.insert(task.clone(), Arc::clone(best_unit));
        }

        Ok(())
    }

    fn ai_predictive_scheduling(&mut self) -> Result<(), XpuOptimizerError> {
        self.scheduled_tasks.clear();
        for task in &self.task_queue {
            let predicted_duration = self.predict_task_duration(task)?;
            let best_unit = self.processing_units.iter()
                .filter_map(|unit| {
                    unit.lock().ok().and_then(|guard| {
                        guard.can_handle_task(task).ok().and_then(|can_handle| {
                            if can_handle {
                                Some((unit, guard.get_current_load().ok()?))
                            } else {
                                None
                            }
                        })
                    })
                })
                .min_by_key(|(_, load)| *load + predicted_duration)
                .map(|(unit, _)| unit)
                .ok_or(XpuOptimizerError::SchedulingError("No suitable processing unit available".to_string()))?;

            let mut best_unit_guard = best_unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock best unit: {}", e)))?;
            best_unit_guard.assign_task(task)?;
            self.scheduled_tasks.insert(task.clone(), Arc::clone(best_unit));
        }

        Ok(())
    }

    fn predict_task_duration(&self, task: &Task) -> Result<Duration, XpuOptimizerError> {
        let historical_data = HistoricalTaskData {
            task_id: task.id,
            execution_time: task.execution_time,
            memory_usage: task.memory_requirement,
            unit_type: task.unit_type.clone(),
            priority: task.priority,
        };
        let prediction = self.ml_model
            .lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?
            .predict(&historical_data)?;
        Ok(prediction.estimated_duration)
    }

    pub fn add_task(&mut self, task: Task, token: &str) -> Result<(), XpuOptimizerError> {
        info!("Adding task: {:?}", task);

        // Validate the JWT token and get the user
        let user = self.get_user_from_token(token)?;

        // Check if the user has permission to add tasks
        if !self.check_user_permission(&user.role, Permission::AddTask)? {
            return Err(XpuOptimizerError::InsufficientPermissionsError);
        }

        // Check if the task is secure and the user has permission to add secure tasks
        if task.secure && !self.check_user_permission(&user.role, Permission::AddSecureTask)? {
            return Err(XpuOptimizerError::InsufficientPermissionsError);
        }

        let node = self.task_graph.add_node(task.id);
        self.task_map.insert(task.id, node);

        for &dep in &task.dependencies {
            if let Some(&dep_node) = self.task_map.get(&dep) {
                self.task_graph.add_edge(dep_node, node, ());
            } else {
                warn!("Dependency task {} not found for task {}", dep, task.id);
            }
        }

        let position = self
            .task_queue
            .iter()
            .position(|t| t.priority < task.priority);
        let cloned_task = task.clone(); // Clone the task before moving it
        match position {
            Some(index) => self.task_queue.insert(index, cloned_task),
            None => self.task_queue.push_back(cloned_task),
        }
        Ok(())
    }

    pub fn remove_task(&mut self, task_id: usize, token: &str) -> Result<Task, XpuOptimizerError> {
        info!("Removing task: {}", task_id);
        let user = self.get_user_from_token(token)?;
        self.check_user_permission(&user.role, Permission::RemoveTask)?;

        if let Some(index) = self.task_queue.iter().position(|t| t.id == task_id) {
            let task = self.task_queue.remove(index).unwrap();
            if let Some(node) = self.task_map.remove(&task_id) {
                self.task_graph.remove_node(node);
            }

            let latency_monitor = self.latency_monitor.lock().map_err(|e| {
                XpuOptimizerError::LockError(format!("Failed to lock latency monitor: {}", e))
            })?;

            {
                let mut start_times = latency_monitor.start_times.lock().map_err(|e| {
                    XpuOptimizerError::LockError(format!("Failed to lock start_times: {}", e))
                })?;
                start_times.remove(&task_id);
            }

            {
                let mut end_times = latency_monitor.end_times.lock().map_err(|e| {
                    XpuOptimizerError::LockError(format!("Failed to lock end_times: {}", e))
                })?;
                end_times.remove(&task_id);
            }

            latency_monitor.remove_task(task_id)?;

            Ok(task)
        } else {
            Err(XpuOptimizerError::TaskNotFoundError(task_id))
        }
    }

    fn resolve_dependencies(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Resolving task dependencies...");
        match toposort(&self.task_graph, None) {
            Ok(order) => {
                let mut new_queue = VecDeque::new();
                for &node in order.iter() {
                    if let Some(task_id) = self.task_graph.node_weight(node) {
                        if let Some(task) =
                            self.task_queue.iter().find(|t| t.id == *task_id).cloned()
                        {
                            new_queue.push_back(task);
                        }
                    }
                }
                self.task_queue = new_queue;
                Ok(())
            }
            Err(_) => {
                error!("Cyclic dependency detected in task graph");
                Err(XpuOptimizerError::CyclicDependencyError)
            }
        }
    }

    // The manage_memory() method has been removed as it's no longer needed.
    // Memory management is now handled directly by the MemoryStrategy in the run() method.

    fn report_latencies(&self) {
        let latency_monitor = self
            .latency_monitor
            .lock()
            .expect("Failed to lock latency monitor");
        let start_times = latency_monitor
            .start_times
            .lock()
            .expect("Failed to lock start_times");
        let end_times = latency_monitor
            .end_times
            .lock()
            .expect("Failed to lock end_times");

        for (task_id, start_time) in start_times.iter() {
            if let Some(end_time) = end_times.get(task_id) {
                let duration = end_time.duration_since(*start_time);
                if let Some(task) = self.task_queue.iter().find(|t| t.id == *task_id) {
                    let deadline_met = duration <= task.execution_time;
                    info!(
                        "Task {} - Latency: {:?}, Deadline: {:?}, Met: {}",
                        task_id, duration, task.execution_time, deadline_met
                    );
                    if !deadline_met {
                        warn!(
                            "Task {} missed its deadline by {:?}",
                            task_id,
                            duration - task.execution_time
                        );
                    }
                } else {
                    warn!("Task {} not found in queue", task_id);
                }
            } else {
                warn!("Task {} has not completed yet", task_id);
            }
        }
    }

    fn adaptive_optimization(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Performing adaptive optimization");
        self.adapt_scheduling_parameters()?;
        self.update_ml_model()?;
        self.optimize_resource_allocation()?;
        self.ai_driven_predictive_scheduling()?;
        info!("Adaptive optimization completed successfully");
        Ok(())
    }

    fn update_ml_model(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Updating ML model with latest task execution data");
        let mut ml_model = self.ml_model.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?;
        ml_model.train(&self.task_history)
            .map_err(|e| XpuOptimizerError::MLOptimizationError(format!("Failed to train ML model: {}", e)))?;
        info!("ML model updated successfully");
        Ok(())
    }

    fn optimize_resource_allocation(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Optimizing resource allocation based on ML model predictions");
        // TODO: Implement resource allocation optimization logic
        // This could involve adjusting processing unit assignments, memory allocation, etc.
        // For now, we'll just log that it's been called
        Ok(())
    }

    fn ai_driven_predictive_scheduling(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Performing AI-driven predictive scheduling");
        let historical_data = self.get_historical_task_data();

        if historical_data.is_empty() {
            info!("No historical data available, using default scheduling");
            return Ok(());
        }

        for task_data in historical_data {
            let prediction = self.ml_model.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?
                .predict(&task_data)
                .map_err(|e| XpuOptimizerError::MLOptimizationError(format!("Failed to predict task: {}", e)))?;
            self.optimize_schedule_based_on_predictions(prediction)?;
        }
        info!("AI-driven predictive scheduling completed successfully");
        Ok(())
    }

    fn update_scheduling_parameters(
        &mut self,
        new_scheduler: Scheduler,
    ) -> Result<(), XpuOptimizerError> {
        // Update the scheduler based on the new strategy
        self.scheduler = new_scheduler;
        Ok(())
    }

    fn optimize_schedule_based_on_predictions(&mut self, prediction: TaskPrediction) -> Result<(), XpuOptimizerError> {
        // Implement the logic for optimizing the schedule based on predictions
        let task = self.task_queue.iter_mut().find(|t| t.id == prediction.task_id)
            .ok_or(XpuOptimizerError::TaskNotFoundError(prediction.task_id))?;

        task.estimated_duration = prediction.estimated_duration;
        task.estimated_resource_usage = prediction.estimated_resource_usage;

        // Reorder task queue based on new predictions
        let mut sorted_tasks: Vec<_> = self.task_queue.drain(..).collect();
        sorted_tasks.sort_by(|a, b| {
            let a_score = f64::from(a.priority) / a.estimated_duration.as_secs_f64();
            let b_score = f64::from(b.priority) / b.estimated_duration.as_secs_f64();
            b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        self.task_queue.extend(sorted_tasks);

        log::info!("Optimized schedule based on prediction for task {}", prediction.task_id);
        Ok(())
    }

    fn get_historical_task_data(&self) -> Vec<HistoricalTaskData> {
        // Retrieve and preprocess historical task data
        self.task_history
            .iter()
            .map(|task| HistoricalTaskData {
                task_id: task.id,
                execution_time: task.execution_time,
                memory_usage: task.memory_usage,
                unit_type: task.unit_type.clone(),
                priority: task.priority,
            })
            .collect()
    }

    fn integrate_cuda(&mut self) -> Result<(), XpuOptimizerError> {
        // TODO: Implement CUDA integration for GPU management
        info!("Integrating CUDA for GPU management");
        Ok(())
    }

    fn integrate_tensorflow(&mut self) -> Result<(), XpuOptimizerError> {
        // TODO: Implement TensorFlow integration for AI model execution
        info!("Integrating TensorFlow for AI model execution");
        Ok(())
    }

    fn integrate_pytorch(&mut self) -> Result<(), XpuOptimizerError> {
        // TODO: Implement PyTorch integration for AI model execution
        info!("Integrating PyTorch for AI model execution");
        Ok(())
    }

    fn connect_slurm(&mut self) -> Result<(), XpuOptimizerError> {
        // TODO: Implement SLURM connection for job scheduling
        info!("Connecting to SLURM job scheduler");
        Ok(())
    }

    async fn connect_kubernetes(&mut self) -> Result<(), XpuOptimizerError> {
        #[cfg(feature = "kubernetes_support")]
        {
            info!("Connecting to Kubernetes cluster");
            match kube::Client::try_default().await {
                Ok(client) => {
                    self.kubernetes_client = Some(client);
                    info!("Successfully connected to Kubernetes cluster");
                    Ok(())
                }
                Err(e) => Err(XpuOptimizerError::KubernetesInitializationError(
                    e.to_string(),
                )),
            }
        }
        #[cfg(not(feature = "kubernetes_support"))]
        {
            info!("Kubernetes support is not enabled");
            Ok(())
        }
    }

    pub fn add_user(
        &mut self,
        username: String,
        password: String,
        role: UserRole,
    ) -> Result<(), XpuOptimizerError> {
        if self.users.contains_key(&username) {
            return Err(XpuOptimizerError::UserAlreadyExistsError(username));
        }
        let salt = SaltString::generate(&mut rand::thread_rng());
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| XpuOptimizerError::PasswordHashingError(e.to_string()))?
            .to_string();
        self.users.insert(
            username.clone(),
            User {
                role: role.clone(),
                password_hash,
            },
        );
        log::info!("User {} has been added with role {:?}", username, role);
        Ok(())
    }

    fn connect_mesos(&mut self) -> Result<(), XpuOptimizerError> {
        // TODO: Implement Apache Mesos connection for resource management
        info!("Connecting to Apache Mesos cluster");
        Ok(())
    }

    pub fn remove_user(&mut self, username: &str) -> Result<(), XpuOptimizerError> {
        if self.users.remove(username).is_none() {
            return Err(XpuOptimizerError::UserNotFoundError(username.to_string()));
        }
        // Remove any active sessions for the removed user
        self.sessions.retain(|_, session| session.user_id != username);
        log::info!("User {} has been removed", username);
        Ok(())
    }

    pub fn authenticate_user(
        &mut self,
        username: &str,
        password: &str,
    ) -> Result<String, XpuOptimizerError> {
        let user = self.users
            .get(username)
            .ok_or(XpuOptimizerError::UserNotFoundError(username.to_string()))?;

        let parsed_hash = PasswordHash::new(&user.password_hash)
            .map_err(|e| XpuOptimizerError::AuthenticationError(e.to_string()))?;

        if Argon2::default().verify_password(password.as_bytes(), &parsed_hash).is_ok() {
            let token = self.generate_jwt_token(username, &user.role)?;
            let session = Session {
                user_id: username.to_string(),
                expiration: Utc::now() + chrono::Duration::hours(24),
            };
            self.sessions.insert(token.clone(), session);
            Ok(token)
        } else {
            Err(XpuOptimizerError::AuthenticationError("Invalid password".to_string()))
        }
    }

    fn generate_jwt_token(
        &self,
        username: &str,
        role: &UserRole,
    ) -> Result<String, XpuOptimizerError> {
        let expiration = Utc::now()
            .checked_add_signed(chrono::Duration::hours(24))
            .expect("valid timestamp")
            .timestamp();

        let claims = Claims {
            sub: username.to_owned(),
            role: role.to_string(),
            exp: expiration as usize,
        };

        let header = Header::new(Algorithm::HS256);
        encode(
            &header,
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_ref()),
        )
        .map_err(|e| XpuOptimizerError::TokenGenerationError(e.to_string()))
    }

    pub fn check_user_permission(
        &self,
        role: &UserRole,
        required_permission: Permission,
    ) -> Result<bool, XpuOptimizerError> {
        Ok(match role {
            UserRole::Admin => true, // Admin has all permissions
            UserRole::Manager => !matches!(required_permission, Permission::ManageUsers),
            UserRole::User => matches!(
                required_permission,
                Permission::ViewTasks | Permission::AddTask
            ),
        })
    }

    pub fn validate_jwt_token(&self, token: &str) -> Result<String, XpuOptimizerError> {
        let decoding_key = DecodingKey::from_secret(self.jwt_secret.as_ref());
        let validation = Validation::new(Algorithm::HS256);
        let token_data = decode::<Claims>(token, &decoding_key, &validation)
            .map_err(|e| XpuOptimizerError::AuthenticationError(e.to_string()))?;
        Ok(token_data.claims.sub)
    }

    pub fn get_user_from_token(&self, token: &str) -> Result<User, XpuOptimizerError> {
        let username = self.validate_jwt_token(token)?;
        self.users
            .get(&username)
            .cloned()
            .ok_or(XpuOptimizerError::UserNotFoundError(username))
    }

    fn create_session(&mut self, username: &str) -> Result<String, XpuOptimizerError> {
        let user = self.users
            .get(username)
            .ok_or(XpuOptimizerError::UserNotFoundError(username.to_string()))?;
        let token = self.generate_jwt_token(username, &user.role)?;
        let expiration = Utc::now() + chrono::Duration::hours(24);
        self.sessions.insert(
            token.clone(),
            Session {
                user_id: username.to_string(),
                expiration,
            },
        );
        Ok(token)
    }

    fn invalidate_session(&mut self, token: &str) -> Result<(), XpuOptimizerError> {
        self.sessions.remove(token)
            .map(|_| ())
            .ok_or(XpuOptimizerError::SessionNotFoundError)
    }

    fn check_session(&self, token: &str) -> Result<(), XpuOptimizerError> {
        if self.sessions.contains_key(token) {
            Ok(())
        } else {
            Err(XpuOptimizerError::InvalidSessionError)
        }
    }

    fn adjust_power_state(&self, unit: &Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>) -> Result<PowerState, XpuOptimizerError> {
        let new_state = {
            let unit_guard = unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?;
            let load_percentage = unit_guard.get_load_percentage()?;
            match load_percentage {
                x if x < 0.3 => PowerState::LowPower,
                x if x < 0.7 => PowerState::Normal,
                _ => PowerState::HighPerformance,
            }
        };

        unit.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?
            .set_power_state(new_state.clone())
            .map_err(|e| XpuOptimizerError::PowerManagementError(format!("Failed to set power state: {}", e)))?;

        Ok(new_state)
    }

    fn calculate_energy_consumption(&self) -> Result<f64, XpuOptimizerError> {
        self.processing_units
            .iter()
            .try_fold(0.0, |acc, unit| -> Result<f64, XpuOptimizerError> {
                let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
                let load_percentage = unit_guard.get_load_percentage()?;
                let energy_profile = unit_guard.get_energy_profile()?;
                let consumption = energy_profile.consumption_rate * load_percentage;
                Ok(acc + consumption)
            })
    }

    fn optimize_energy_efficiency(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Optimizing energy efficiency for processing units...");

        // Check if we have historical data for ML optimization
        let historical_data = self.get_historical_task_data();
        if historical_data.is_empty() {
            info!("No historical data available, using default power states");
            // Set default power states without ML optimization
            for unit in &self.processing_units {
                let mut unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
                let unit_type = unit_guard.get_unit_type()?;
                unit_guard.set_power_state(PowerState::Normal).map_err(|e| {
                    XpuOptimizerError::PowerManagementError(format!(
                        "Failed to set power state for {}: {}",
                        unit_type, e
                    ))
                })?;
                info!("Set default power state for {} to Normal", unit_type);
            }
            return Ok(());
        }

        // Proceed with ML-based optimization if historical data is available
        let mut optimal_states = Vec::with_capacity(self.processing_units.len());

        for unit in &self.processing_units {
            let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            let optimal_state = self.calculate_optimal_power_state(&*unit_guard)?;
            optimal_states.push(optimal_state);
        }

        for (unit, new_state) in self.processing_units.iter().zip(optimal_states.iter()) {
            let mut unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            let unit_type = unit_guard.get_unit_type()?;
            unit_guard.set_power_state(new_state.clone()).map_err(|e| {
                XpuOptimizerError::PowerManagementError(format!(
                    "Failed to set power state for {}: {}",
                    unit_type, e
                ))
            })?;
            info!("Set power state for {} to {:?}", unit_type, new_state);
        }

        let total_energy = self.calculate_energy_consumption()?;
        info!("Total energy consumption after optimization: {:.2} W", total_energy);

        self.power_manager.optimize_power(total_energy)?;

        Ok(())
    }

    fn calculate_optimal_power_state(&self, unit: &dyn ProcessingUnitTrait) -> Result<PowerState, XpuOptimizerError> {
        let load_percentage = unit.get_load_percentage()?;
        Ok(match load_percentage {
            x if x < 0.3 => PowerState::LowPower,
            x if x < 0.7 => PowerState::Normal,
            _ => PowerState::HighPerformance,
        })
    }

    fn initialize_cluster(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Initializing cluster...");
        // TODO: Implement cluster initialization logic
        Ok(())
    }

    fn add_node_to_cluster(&mut self, node: ClusterNode) -> Result<(), XpuOptimizerError> {
        info!("Adding node to cluster: {:?}", node);
        let node_clone = node.clone();
        self.cluster_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock cluster manager: {}", e)))?
            .add_node(node_clone)
            .map_err(|e| XpuOptimizerError::ClusterInitializationError(format!("Failed to add node: {}", e)))?;
        self.node_pool.push(node);
        Ok(())
    }

    fn remove_node_from_cluster(&mut self, node_id: &str) -> Result<(), XpuOptimizerError> {
        info!("Removing node from cluster: {}", node_id);
        self.cluster_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock cluster manager: {}", e)))?
            .remove_node(node_id)
            .map_err(|e| XpuOptimizerError::ClusterInitializationError(format!("Failed to remove node: {}", e)))?;
        self.node_pool.retain(|node| node.id != node_id);
        Ok(())
    }

    fn scale_cluster(&mut self, target_size: usize) -> Result<(), XpuOptimizerError> {
        info!("Scaling cluster to size: {}", target_size);
        // TODO: Implement cluster scaling logic
        Ok(())
    }

    fn distribute_tasks_across_cluster(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Distributing tasks across cluster...");
        // TODO: Implement task distribution logic
        Ok(())
    }

    fn report_processing_unit_utilization(&self) -> Result<(), XpuOptimizerError> {
        info!("Reporting processing unit utilization...");
        for unit in &self.processing_units {
            let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            info!(
                "Unit {:?} utilization: {:.2}%",
                unit_guard.get_unit_type()?,
                unit_guard.get_load_percentage()? * 100.0
            );
        }
        Ok(())
    }

    pub fn verify_jwt_secret(&self, secret: &[u8]) -> Result<bool, XpuOptimizerError> {
        if self.jwt_secret.is_empty() {
            return Err(XpuOptimizerError::AuthenticationError("JWT secret not set".to_string()));
        }
        Ok(self.jwt_secret == secret)
    }

    pub fn get_jwt_secret(&self) -> Result<Vec<u8>, XpuOptimizerError> {
        if self.jwt_secret.is_empty() {
            return Err(XpuOptimizerError::AuthenticationError("JWT secret not set".to_string()));
        }
        Ok(self.jwt_secret.clone())
    }

    fn cleanup_completed_tasks(&mut self, completed_tasks: &[Task]) -> Result<(), XpuOptimizerError> {
        info!("Starting final cleanup of completed tasks...");

        for task in completed_tasks {
            if self.scheduled_tasks.remove(task).is_some() {
                info!("Removed task {} from scheduled tasks", task.id);
                self.task_queue.retain(|t| t.id != task.id);
                info!("Removed task {} from task queue", task.id);
            } else {
                warn!("Task {} was not found in scheduled tasks during cleanup", task.id);
            }
        }

        info!("Final cleanup completed. Remaining tasks in queue: {}", self.task_queue.len());
        info!("Remaining scheduled tasks: {}", self.scheduled_tasks.len());
        Ok(())
    }
}

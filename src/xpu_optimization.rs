use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::cmp::Ordering;

use argon2::{
    password_hash::SaltString, Argon2, PasswordHash, PasswordHasher, PasswordVerifier,
};
use chrono::Utc;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};
use log::{info, warn, error, debug};

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

impl ToString for UserRole {
    fn to_string(&self) -> String {
        match self {
            UserRole::Admin => "Admin".to_string(),
            UserRole::Manager => "Manager".to_string(),
            UserRole::User => "User".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct User {
    pub role: UserRole,
    pub password_hash: String,
}

pub struct Role {
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
    pub scheduled_tasks: Vec<Task>,
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
        let unit_types = [
            ProcessingUnitType::CPU,
            ProcessingUnitType::GPU,
            ProcessingUnitType::TPU,
            ProcessingUnitType::NPU,
            ProcessingUnitType::LPU,
            ProcessingUnitType::VPU,
            ProcessingUnitType::FPGA,
        ];
        for i in 0..num {
            let unit_type = &unit_types[i % unit_types.len()];
            let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = match unit_type {
                ProcessingUnitType::CPU => Arc::new(Mutex::new(CPU::new(i, 1.0))),
                ProcessingUnitType::GPU => Arc::new(Mutex::new(GPU::new(i, 1.0))),
                ProcessingUnitType::TPU => Arc::new(Mutex::new(TPU::new(i, 1.0))),
                ProcessingUnitType::NPU => Arc::new(Mutex::new(NPU::new(i, 1.0))),
                ProcessingUnitType::LPU => Arc::new(Mutex::new(LPU::new(i, 1.0))),
                ProcessingUnitType::VPU => Arc::new(Mutex::new(VPU::new(i, 1.0))),
                ProcessingUnitType::FPGA => Arc::new(Mutex::new(FPGACore::new(i, 1.0))),
            };
            self.processing_units.push(unit);
        }
        info!("Initialized {} processing units", self.processing_units.len());
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
        self.config.cloud_offloading_policy = policy.clone();
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
}



impl XpuOptimizer {
    fn adapt_scheduling_parameters(&mut self) -> Result<(), XpuOptimizerError> {
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
        let mut distribution = HashMap::new();
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
                .or_insert_with(Vec::new)
                .push(task.clone());
        }

        Ok(distribution)
    }
}



use crate::cloud_offloading::DefaultCloudOffloader;

impl XpuOptimizer {
    pub fn new(config: XpuOptimizerConfig) -> Result<Self, XpuOptimizerError> {
        info!("Initializing XpuOptimizer with custom configuration");
        let mut processing_units: Vec<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>> = Vec::new();

        let unit_types = [
            ProcessingUnitType::CPU,
            ProcessingUnitType::GPU,
            ProcessingUnitType::TPU,
            ProcessingUnitType::NPU,
            ProcessingUnitType::LPU,
            ProcessingUnitType::VPU,
            ProcessingUnitType::FPGA,
        ];

        let default_processing_power = 1.0;

        // Ensure at least one of each processing unit type is created
        for (i, unit_type) in unit_types.iter().enumerate() {
            let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = match unit_type {
                ProcessingUnitType::CPU => Arc::new(Mutex::new(CPU::new(i, default_processing_power))),
                ProcessingUnitType::GPU => Arc::new(Mutex::new(GPU::new(i, default_processing_power))),
                ProcessingUnitType::TPU => Arc::new(Mutex::new(TPU::new(i, default_processing_power))),
                ProcessingUnitType::NPU => Arc::new(Mutex::new(NPU::new(i, default_processing_power))),
                ProcessingUnitType::LPU => Arc::new(Mutex::new(LPU::new(i, default_processing_power))),
                ProcessingUnitType::VPU => Arc::new(Mutex::new(VPU::new(i, default_processing_power))),
                ProcessingUnitType::FPGA => Arc::new(Mutex::new(FPGACore::new(i, default_processing_power))),
            };
            processing_units.push(unit);
        }

        // Add remaining processing units to reach the configured number
        for i in unit_types.len()..config.num_processing_units {
            let unit_type = &unit_types[i % unit_types.len()];
            let unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>> = match unit_type {
                ProcessingUnitType::CPU => Arc::new(Mutex::new(CPU::new(i, default_processing_power))),
                ProcessingUnitType::GPU => Arc::new(Mutex::new(GPU::new(i, default_processing_power))),
                ProcessingUnitType::TPU => Arc::new(Mutex::new(TPU::new(i, default_processing_power))),
                ProcessingUnitType::NPU => Arc::new(Mutex::new(NPU::new(i, default_processing_power))),
                ProcessingUnitType::LPU => Arc::new(Mutex::new(LPU::new(i, default_processing_power))),
                ProcessingUnitType::VPU => Arc::new(Mutex::new(VPU::new(i, default_processing_power))),
                ProcessingUnitType::FPGA => Arc::new(Mutex::new(FPGACore::new(i, default_processing_power))),
            };
            processing_units.push(unit);
        }

        info!("Created {} processing units", processing_units.len());

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
            scheduled_tasks: Vec::new(),
        })
    }

    pub fn run(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Running XPU optimization...");
        let start_time = Instant::now();

        self.resolve_dependencies()?;
        self.allocate_memory_for_tasks()?;
        self.schedule_tasks()?;
        self.optimize_energy_efficiency()?;
        self.execute_tasks()?;

        let total_duration = start_time.elapsed();
        info!("XPU optimization completed in {:?}", total_duration);

        self.report_metrics()?;
        self.adaptive_optimization()?;
        self.deallocate_memory_for_completed_tasks()?;
        self.update_system_status()?;

        info!("XPU optimization run completed successfully");
        Ok(())
    }

    pub fn allocate_memory_for_tasks(&mut self) -> Result<(), XpuOptimizerError> {
        let mut memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;
        memory_manager.allocate_for_tasks(self.task_queue.make_contiguous())
            .map_err(|e| XpuOptimizerError::MemoryError(format!("Failed to allocate memory for tasks: {}", e)))
    }

    fn deallocate_memory_for_completed_tasks(&mut self) -> Result<(), XpuOptimizerError> {
        let mut memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;
        memory_manager.deallocate_completed_tasks(&self.scheduled_tasks)
            .map_err(|e| XpuOptimizerError::MemoryError(format!("Failed to deallocate memory for completed tasks: {}", e)))
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

    fn execute_tasks(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Executing tasks on respective processing units...");

        // Sort tasks by priority (highest to lowest)
        self.scheduled_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut execution_results = Vec::new();
        let total_memory = self.config.memory_pool_size;
        let mut available_memory = self.get_available_memory()?;

        for task in self.scheduled_tasks.clone() {
            info!("Attempting to execute task {} (priority: {}, memory requirement: {}, unit type: {:?})",
                  task.id, task.priority, task.memory_requirement, task.unit_type);

            if task.memory_requirement > total_memory {
                let err = XpuOptimizerError::MemoryError(
                    format!("Memory requirement for task {} ({}) exceeds total memory ({})",
                            task.id, task.memory_requirement, total_memory)
                );
                error!("{}", err);
                execution_results.push(Err(err));
                continue;
            }

            // Check if there's enough available memory and try to free if necessary
            if task.memory_requirement > available_memory {
                match self.try_free_memory(task.memory_requirement - available_memory) {
                    Ok(_) => {
                        available_memory = self.get_available_memory()?;
                        info!("Successfully freed up memory for task {}. Available memory: {}", task.id, available_memory);
                    },
                    Err(e) => {
                        error!("Failed to free up memory for task {}: {}", task.id, e);
                        execution_results.push(Err(e));
                        continue;
                    }
                }
            }

            // Try to allocate memory for the task
            match self.allocate_memory_for_task(&task) {
                Ok(_) => {
                    available_memory -= task.memory_requirement;
                    debug!("Allocated {} memory for task {}. Available memory: {}",
                           task.memory_requirement, task.id, available_memory);
                },
                Err(e) => {
                    error!("Failed to allocate memory for task {}: {}", task.id, e);
                    execution_results.push(Err(e));
                    continue;
                }
            }

            let execution_result = match self.find_suitable_unit(&task) {
                Ok(Some(unit)) => self.execute_task_on_unit(&task, unit),
                Ok(None) => Err(XpuOptimizerError::ProcessingUnitNotFound(format!("No suitable unit found for task {}", task.id))),
                Err(e) => Err(e),
            };

            match &execution_result {
                Ok((id, duration)) => {
                    info!("Task {} executed successfully in {:?}", id, duration);
                },
                Err(e) => {
                    error!("Failed to execute task {}: {}", task.id, e);
                    self.handle_task_execution_error(&task, e);
                }
            }

            execution_results.push(execution_result);

            // Always attempt to deallocate memory after task execution
            match self.deallocate_memory_for_task(&task) {
                Ok(_) => {
                    available_memory += task.memory_requirement;
                    debug!("Successfully deallocated {} memory for task {}. Available memory: {}",
                           task.memory_requirement, task.id, available_memory);
                },
                Err(e) => {
                    error!("Failed to deallocate memory for task {}: {}", task.id, e);
                    if let Err(free_err) = self.try_free_memory(task.memory_requirement) {
                        error!("Failed to free memory for task {}: {}. Memory leak may occur!", task.id, free_err);
                    } else {
                        warn!("Forcefully freed {} memory for task {}", task.memory_requirement, task.id);
                        available_memory += task.memory_requirement;
                    }
                }
            }

            available_memory = self.get_available_memory()?;
            debug!("Available memory after task {}: {}", task.id, available_memory);
        }

        self.process_execution_results(execution_results)
    }

    fn execute_task_with_memory_management(&mut self, task: &Task, available_memory: &mut usize, unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>) -> Result<(u32, Duration), XpuOptimizerError> {
        if task.memory_requirement > *available_memory {
            self.try_free_memory(task.memory_requirement - *available_memory)?;
            *available_memory = self.get_available_memory()?;

            if task.memory_requirement > *available_memory {
                return Err(XpuOptimizerError::MemoryError(
                    format!("Insufficient memory for task {} after freeing: required {}, available {}",
                            task.id, task.memory_requirement, available_memory)
                ));
            }
        }

        self.allocate_memory_for_task(task)?;
        *available_memory -= task.memory_requirement;
        info!("Memory allocated for task {}. Remaining memory: {}", task.id, available_memory);

        let result = self.execute_task_on_unit(task, unit)?;

        match self.deallocate_memory_for_task(task) {
            Ok(_) => {
                *available_memory += task.memory_requirement;
                info!("Successfully deallocated memory for task {}. Available memory: {}", task.id, available_memory);
            },
            Err(e) => {
                error!("Failed to deallocate memory for task {}: {}. Attempting to free memory...", task.id, e);
                if let Err(free_err) = self.try_free_memory(task.memory_requirement) {
                    error!("Failed to free memory for task {}: {}. Memory leak may occur!", task.id, free_err);
                } else {
                    *available_memory += task.memory_requirement;
                    info!("Successfully freed memory for task {}. Available memory: {}", task.id, available_memory);
                }
            }
        }

        Ok(result)
    }

    pub fn get_available_memory(&self) -> Result<usize, XpuOptimizerError> {
        let memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;

        Ok(memory_manager.get_available_memory())
    }

    fn try_free_memory(&mut self, required_memory: usize) -> Result<(), XpuOptimizerError> {
        let mut memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;

        let available_memory = memory_manager.get_available_memory();
        if available_memory >= required_memory {
            return Ok(());
        }

        // Attempt to free memory by deallocating from low-priority tasks
        let mut freed_memory = 0;
        for task in self.scheduled_tasks.iter().rev() {
            if freed_memory >= required_memory - available_memory {
                break;
            }
            if let Err(e) = memory_manager.deallocate(task.memory_requirement) {
                warn!("Failed to deallocate memory for task {}: {}", task.id, e);
            } else {
                freed_memory += task.memory_requirement;
            }
        }

        if memory_manager.get_available_memory() >= required_memory {
            Ok(())
        } else {
            Err(XpuOptimizerError::MemoryError(format!(
                "Failed to free enough memory. Required: {}, Available: {}",
                required_memory,
                memory_manager.get_available_memory()
            )))
        }
    }

    fn handle_task_execution_error(&self, task: &Task, error: &XpuOptimizerError) {
        match error {
            XpuOptimizerError::MemoryError(_) => {
                warn!("Memory allocation failed for task {}. Attempting to free up memory.", task.id);
                if let Err(e) = self.deallocate_memory_for_task(task) {
                    error!("Failed to deallocate memory for task {}: {}", task.id, e);
                }
            },
            XpuOptimizerError::ProcessingUnitNotFound(_) => {
                warn!("No suitable processing unit found for task {}. Skipping task.", task.id);
            },
            _ => {
                warn!("Unexpected error occurred while executing task {}. Skipping task.", task.id);
            }
        }
    }

    fn find_and_execute_task(&mut self, task: &Task, available_memory: &mut usize) -> Result<(u32, Duration), XpuOptimizerError> {
        // Find a suitable processing unit for the task
        let suitable_unit = self.find_suitable_unit(task)?
            .ok_or_else(|| XpuOptimizerError::ProcessingUnitNotFound(format!("No suitable unit found for task {} of type {:?}", task.id, task.unit_type)))?;

        // Check if there's enough memory available
        if task.memory_requirement > *available_memory {
            return Err(XpuOptimizerError::MemoryError(format!("Insufficient memory for task {}: required {}, available {}", task.id, task.memory_requirement, available_memory)));
        }

        // Allocate memory for the task
        self.allocate_memory_for_task(task)?;
        *available_memory -= task.memory_requirement;
        info!("Memory allocated for task {}: {} units", task.id, task.memory_requirement);

        // Execute the task
        let result = {
            let mut unit_guard = suitable_unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?;
            unit_guard.execute_task(task)
        };

        // Handle the result
        match result {
            Ok(duration) => {
                info!("Task {} executed successfully in {:?}", task.id, duration);
                Ok((task.id as u32, duration))
            },
            Err(e) => {
                error!("Task {} failed: {}", task.id, e);
                Err(e)
            }
        }
    }

    fn allocate_memory_for_task(&self, task: &Task) -> Result<(), XpuOptimizerError> {
        let mut memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;

        let available_memory = memory_manager.get_available_memory();
        if available_memory < task.memory_requirement {
            error!("Insufficient memory for task {}: required {}, available {}",
                   task.id, task.memory_requirement, available_memory);
            return Err(XpuOptimizerError::MemoryError(
                format!("Insufficient memory for task {}: required {}, available {}",
                        task.id, task.memory_requirement, available_memory)
            ));
        }

        if task.memory_requirement == 0 {
            warn!("Task {} requires 0 memory, skipping allocation", task.id);
            return Ok(());
        }

        info!("Attempting to allocate {} memory units for task {}", task.memory_requirement, task.id);
        match memory_manager.allocate(task.memory_requirement) {
            Ok(_) => {
                info!("Successfully allocated {} memory units for task {}", task.memory_requirement, task.id);
                Ok(())
            },
            Err(e) => {
                error!("Failed to allocate memory for task {}: {}", task.id, e);
                // Check if any memory was partially allocated
                let new_available_memory = memory_manager.get_available_memory();
                if new_available_memory < available_memory {
                    let partially_allocated = available_memory - new_available_memory;
                    warn!("Detected partial allocation of {} memory units for task {}", partially_allocated, task.id);
                    match memory_manager.deallocate(partially_allocated) {
                        Ok(_) => info!("Successfully deallocated partially allocated memory for task {}", task.id),
                        Err(dealloc_err) => error!("Failed to deallocate partially allocated memory for task {}: {}", task.id, dealloc_err),
                    }
                }
                Err(XpuOptimizerError::MemoryError(format!("Failed to allocate {} memory units for task {}: {}", task.memory_requirement, task.id, e)))
            }
        }
    }

    fn deallocate_memory_for_task(&self, task: &Task) -> Result<(), XpuOptimizerError> {
        let mut memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;

        if task.memory_requirement == 0 {
            debug!("Task {} requires no memory, skipping deallocation", task.id);
            return Ok(());
        }

        match memory_manager.deallocate(task.memory_requirement) {
            Ok(_) => {
                info!("Successfully deallocated {} memory units for task {}", task.memory_requirement, task.id);
                Ok(())
            },
            Err(e) => {
                error!("Failed to deallocate memory for task {}: {}", task.id, e);
                // Attempt to recover by forcefully freeing the memory
                match memory_manager.force_free(task.memory_requirement) {
                    Ok(_) => {
                        warn!("Forcefully freed {} memory units for task {}", task.memory_requirement, task.id);
                        Ok(()) // Return Ok since we managed to free the memory
                    },
                    Err(force_free_err) => {
                        error!("Failed to forcefully free memory for task {}: {}", task.id, force_free_err);
                        // Attempt partial deallocation
                        let partial_deallocation = memory_manager.get_available_memory();
                        if partial_deallocation > 0 {
                            warn!("Partially deallocated {} memory units for task {}", partial_deallocation, task.id);
                            Ok(())
                        } else {
                            Err(XpuOptimizerError::MemoryError(format!(
                                "Failed to deallocate and forcefully free memory for task {}: {}",
                                task.id, force_free_err
                            )))
                        }
                    }
                }
            }
        }
    }

    fn process_execution_results(&mut self, execution_results: Vec<Result<(u32, Duration), XpuOptimizerError>>) -> Result<(), XpuOptimizerError> {
        let (successful_tasks, failed_tasks): (Vec<_>, Vec<_>) = execution_results.into_iter()
            .partition(|result| result.is_ok());

        let successful_count = successful_tasks.len();
        let failed_count = failed_tasks.len();

        for result in successful_tasks {
            if let Ok((task_id, duration)) = result {
                info!("Task {} completed successfully in {:?}", task_id, duration);
                match self.update_task_history(task_id as usize, duration, true) {
                    Ok(_) => info!("Task history updated for successful task {}", task_id),
                    Err(e) => warn!("Failed to update task history for successful task {}: {}", task_id, e),
                }
            }
        }

        let mut error_summary = Vec::new();
        for result in failed_tasks {
            if let Err(ref e) = result {
                error!("Error executing task: {}", e);
                let error_type = match e {
                    XpuOptimizerError::MemoryError(ref msg) => {
                        error!("Memory allocation error: {}", msg);
                        "Memory allocation"
                    },
                    XpuOptimizerError::TaskExecutionError(ref msg) => {
                        error!("Task execution error: {}", msg);
                        "Task execution"
                    },
                    _ => {
                        error!("Unexpected error type: {:?}", e);
                        "Unexpected"
                    }
                };
                if let Some(task_id) = e.task_id() {
                    match self.update_task_history(task_id as usize, Duration::from_secs(0), false) {
                        Ok(_) => info!("Task history updated for failed task {}", task_id),
                        Err(update_err) => warn!("Failed to update task history for failed task {}: {}", task_id, update_err),
                    }
                    error_summary.push(format!("Task {}: {} error", task_id, error_type));
                } else {
                    warn!("Could not update task history for failed task: task ID unknown");
                    error_summary.push(format!("Unknown task: {} error", error_type));
                }
            }
        }

        info!("Task execution completed. Successful: {}, Failed: {}", successful_count, failed_count);

        if failed_count > 0 {
            warn!("Some tasks failed during execution. Error summary: {}", error_summary.join(", "));
            Err(XpuOptimizerError::TaskExecutionError(format!("{} out of {} tasks failed. Errors: {}",
                failed_count, successful_count + failed_count, error_summary.join(", "))))
        } else {
            info!("All tasks completed successfully");
            Ok(())
        }
    }

    fn execute_single_task(&mut self, task: &Task) -> Result<(u32, Duration), XpuOptimizerError> {
        let available_memory = self.get_available_memory()?;

        if available_memory < task.memory_requirement {
            return Err(XpuOptimizerError::MemoryError(
                format!("Insufficient memory for task {}: required {}, available {}",
                    task.id, task.memory_requirement, available_memory)
            ));
        }

        // Allocate memory for the task
        self.allocate_memory_for_task(task).map_err(|e| {
            error!("Failed to allocate memory for task {}: {}", task.id, e);
            e
        })?;
        info!("Memory allocated for task {}: {} units", task.id, task.memory_requirement);

        let result = self.find_suitable_unit(task)
            .and_then(|opt_unit| opt_unit.ok_or_else(|| XpuOptimizerError::ProcessingUnitNotFound(
                format!("No suitable unit found for task {} of type {:?}", task.id, task.unit_type)
            )))
            .and_then(|unit| self.execute_task_on_unit(task, unit));

        // Handle the result
        match &result {
            Ok((_, duration)) => {
                info!("Task {} executed successfully in {:?}", task.id, duration);
                self.update_task_history(task.id, *duration, true).map_err(|e| {
                    error!("Failed to update task history for task {}: {}", task.id, e);
                    e
                })?;
            },
            Err(e) => {
                error!("Failed to execute task {}: {}", task.id, e);
                self.update_task_history(task.id, Duration::from_secs(0), false).map_err(|e| {
                    error!("Failed to update task history for failed task {}: {}", task.id, e);
                    e
                })?;
            }
        }

        // Always attempt to deallocate memory, even if task execution failed
        self.deallocate_memory_for_task(task).map_err(|e| {
            error!("Failed to deallocate memory for task {}: {}", task.id, e);
            e
        })?;

        result
    }

    fn schedule_tasks(&mut self) -> Result<(), XpuOptimizerError> {
        info!("Scheduling tasks with pluggable strategy and adaptive optimization...");
        self.resolve_dependencies()?;

        let tasks: Vec<Task> = self.task_queue.iter().cloned().collect();
        info!("Total tasks to schedule: {}", tasks.len());

        let mut scheduled_tasks = Vec::new();
        let mut unscheduled_tasks = Vec::new();

        // Sort tasks by priority (highest to lowest) and then by memory requirement (lowest to highest)
        let mut sorted_tasks = tasks;
        sorted_tasks.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.memory_requirement.cmp(&b.memory_requirement)));

        let total_memory = self.get_available_memory()?;
        let mut remaining_memory = total_memory;

        for task in sorted_tasks {
            info!("Attempting to schedule task {} (priority: {}, memory: {})", task.id, task.priority, task.memory_requirement);
            if task.memory_requirement <= remaining_memory {
                match self.schedule_single_task(&task, 3) {
                    Ok(unit) => {
                        match self.allocate_memory_for_task(&task) {
                            Ok(_) => {
                                scheduled_tasks.push((task.clone(), unit));
                                remaining_memory -= task.memory_requirement;
                                info!("Task {} (priority: {}, memory: {}) scheduled successfully on {:?}",
                                      task.id, task.priority, task.memory_requirement, unit.lock().unwrap().get_unit_type());
                            },
                            Err(e) => {
                                warn!("Failed to allocate memory for task {} (priority: {}, memory: {}): {}",
                                      task.id, task.priority, task.memory_requirement, e);
                                unscheduled_tasks.push(task);
                            }
                        }
                    },
                    Err(e) => {
                        warn!("Failed to schedule task {} (priority: {}, memory: {}): {}",
                              task.id, task.priority, task.memory_requirement, e);
                        unscheduled_tasks.push(task);
                    }
                }
            } else {
                warn!("Insufficient memory for task {} (priority: {}, memory: {}). Available: {}",
                      task.id, task.priority, task.memory_requirement, remaining_memory);
                unscheduled_tasks.push(task);
            }
        }

        info!("Initially scheduled tasks: {}, Unscheduled tasks: {}", scheduled_tasks.len(), unscheduled_tasks.len());
        self.log_unscheduled_tasks(&unscheduled_tasks);

        if !unscheduled_tasks.is_empty() {
            info!("Attempting to reschedule {} unscheduled tasks", unscheduled_tasks.len());
            self.reschedule_tasks(&mut unscheduled_tasks, &mut scheduled_tasks)?;
        }

        self.update_scheduled_tasks(scheduled_tasks)?;

        if !unscheduled_tasks.is_empty() {
            warn!("Failed to schedule {} tasks", unscheduled_tasks.len());
            self.handle_unscheduled_tasks(&unscheduled_tasks)?;
        }

        info!("All {} tasks successfully scheduled or handled", self.scheduled_tasks.len());

        self.optimize_energy_efficiency()?;
        self.adapt_scheduling_parameters()?;

        info!("Tasks scheduled with pluggable strategy and adaptive optimization");
        Ok(())
    }

    fn schedule_single_task(&self, task: &Task, max_attempts: usize) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        for attempt in 1..=max_attempts {
            match self.find_suitable_unit(task) {
                Ok(Some(unit)) => return Ok(unit),
                Ok(None) if attempt < max_attempts => {
                    warn!("Scheduling attempt {} for task {} failed. Retrying...", attempt, task.id);
                    std::thread::sleep(std::time::Duration::from_millis(100 * attempt as u64));
                },
                Ok(None) => {
                    if attempt == max_attempts {
                        return Err(XpuOptimizerError::SchedulingError(format!("No suitable unit found for task {} after {} attempts", task.id, max_attempts)));
                    }
                },
                Err(e) => return Err(e),
            }
        }
        unreachable!("This line should never be reached due to the return in the last iteration of the loop")
    }

    fn log_unscheduled_tasks(&self, unscheduled_tasks: &[Task]) {
        if !unscheduled_tasks.is_empty() {
            warn!("Not all tasks were initially scheduled. Unscheduled: {}", unscheduled_tasks.len());
            for task in unscheduled_tasks {
                warn!("Task {} (type: {:?}, priority: {}) could not be initially scheduled", task.id, task.unit_type, task.priority);
            }
        }
    }

    fn update_scheduled_tasks(&mut self, scheduled_tasks: Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>) -> Result<(), XpuOptimizerError> {
        self.scheduled_tasks = scheduled_tasks.into_iter().map(|(task, _)| task).collect();
        info!("Updated scheduled tasks. Total scheduled: {}", self.scheduled_tasks.len());
        Ok(())
    }

    fn reschedule_tasks(&mut self, unscheduled_tasks: &mut Vec<Task>, scheduled_tasks: &mut Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>) -> Result<(), XpuOptimizerError> {
        info!("Attempting to reschedule {} unscheduled tasks", unscheduled_tasks.len());

        unscheduled_tasks.retain(|task| {
            match self.find_suitable_unit_relaxed(task) {
                Ok(Some(unit)) => {
                    scheduled_tasks.push((task.clone(), unit));
                    info!("Rescheduled task {} successfully with relaxed constraints", task.id);
                    false // Remove from unscheduled_tasks
                },
                _ => {
                    warn!("Failed to reschedule task {} even with relaxed constraints", task.id);
                    true // Keep in unscheduled_tasks
                }
            }
        });

        if !unscheduled_tasks.is_empty() {
            info!("Attempting fallback strategy for {} remaining tasks", unscheduled_tasks.len());
            self.fallback_strategy(unscheduled_tasks)?;
        }

        Ok(())
    }

    fn fallback_strategy(&mut self, unscheduled_tasks: &mut Vec<Task>) -> Result<(), XpuOptimizerError> {
        let mut offloaded_tasks = Vec::new();
        let mut failed_tasks = Vec::new();

        for task in unscheduled_tasks.drain(..) {
            match self.offload_to_cloud(&task) {
                Ok(_) => {
                    info!("Task {} offloaded to cloud", task.id);
                    offloaded_tasks.push(task);
                },
                Err(e) => {
                    error!("Failed to offload task {} to cloud: {}", task.id, e);
                    failed_tasks.push(task);
                }
            }
        }

        if !failed_tasks.is_empty() {
            warn!("Failed to schedule or offload {} tasks", failed_tasks.len());
            unscheduled_tasks.extend(failed_tasks);
        }

        self.update_offloaded_tasks(offloaded_tasks);
        Ok(())
    }

    fn handle_unscheduled_tasks(&self, unscheduled_tasks: &[Task]) -> Result<(), XpuOptimizerError> {
        for task in unscheduled_tasks {
            error!("Task {} (type: {:?}, priority: {}) could not be scheduled or offloaded", task.id, task.unit_type, task.priority);
        }
        // Depending on your error handling strategy, you might want to return an error here
        // or implement a mechanism to retry these tasks later
        Ok(())
    }

    fn update_offloaded_tasks(&mut self, offloaded_tasks: Vec<Task>) {
        // Implement logic to track offloaded tasks, if necessary
        info!("Updated offloaded tasks. Total offloaded: {}", offloaded_tasks.len());
    }

    fn find_suitable_unit_relaxed(&self, task: &Task) -> Result<Option<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>>, XpuOptimizerError> {
        let mut best_unit = None;
        let mut lowest_load = f64::MAX;

        for unit in &self.processing_units {
            let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            let unit_type = unit_guard.get_unit_type().map_err(|e| XpuOptimizerError::ProcessingUnitError(e.to_string()))?;
            let load_percentage = unit_guard.get_load_percentage().map_err(|e| XpuOptimizerError::ProcessingUnitError(e.to_string()))?;

            if &unit_type == &task.unit_type || load_percentage < 0.8 {
                if load_percentage < lowest_load {
                    lowest_load = load_percentage;
                    best_unit = Some(Arc::clone(unit));
                }
            }
        }

        Ok(best_unit)
    }

    fn find_suitable_unit(&self, task: &Task) -> Result<Option<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>>, XpuOptimizerError> {
        let mut best_unit = None;
        let mut lowest_score = f64::MAX;

        info!("Searching for suitable unit for task {} of type {:?}", task.id, task.unit_type);

        let mut fallback_units = Vec::new();

        for (index, unit) in self.processing_units.iter().enumerate() {
            let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit {}: {}", index, e)))?;
            let unit_type = unit_guard.get_unit_type().map_err(|e| XpuOptimizerError::ProcessingUnitError(format!("Failed to get unit type for unit {}: {}", index, e)))?;

            match unit_guard.can_handle_task(task) {
                Ok(can_handle) => {
                    if can_handle {
                        let load_percentage = unit_guard.get_load_percentage().map_err(|e| XpuOptimizerError::ProcessingUnitError(format!("Failed to get load percentage for unit {}: {}", index, e)))?;
                        let performance_score = unit_guard.get_performance_score().map_err(|e| XpuOptimizerError::ProcessingUnitError(format!("Failed to get performance score for unit {}: {}", index, e)))?;

                        // Calculate a score based on load, performance, and memory availability
                        let memory_availability = unit_guard.get_available_memory().map_err(|e| XpuOptimizerError::ProcessingUnitError(format!("Failed to get available memory for unit {}: {}", index, e)))?;
                        let score = (load_percentage + (task.memory_requirement as f64 / memory_availability as f64)) / performance_score;

                        if &unit_type == &task.unit_type {
                            if score < lowest_score {
                                lowest_score = score;
                                best_unit = Some(Arc::clone(unit));
                                info!("Found potential unit {} of type {:?} with load {:.2}%, memory {}/{}, and score {:.2} for task {}",
                                      index, unit_type, load_percentage * 100.0, task.memory_requirement, memory_availability, score, task.id);
                            }
                        } else {
                            fallback_units.push((Arc::clone(unit), score, unit_type));
                            debug!("Added fallback unit {} of type {:?} with score {:.2} for task {}", index, unit_type, score, task.id);
                        }
                    } else {
                        debug!("Processing unit {} of type {:?} cannot handle task {} due to insufficient resources", index, unit_type, task.id);
                    }
                },
                Err(e) => warn!("Failed to check if unit {} can handle task {}: {}", index, task.id, e),
            }
        }

        if best_unit.is_none() {
            info!("No exact match found for task {} of type {:?}. Trying fallback units.", task.id, task.unit_type);
            fallback_units.sort_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            best_unit = fallback_units.first().map(|(unit, score, unit_type)| {
                info!("Selected fallback unit of type {:?} for task {} with score {:.2}", unit_type, task.id, score);
                Arc::clone(unit)
            });
        }

        match &best_unit {
            Some(unit) => {
                let unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock selected unit: {}", e)))?;
                let selected_type = unit_guard.get_unit_type()?;
                info!("Selected unit for task {} of type {:?}: {:?} with score {:.2}", task.id, task.unit_type, selected_type, lowest_score);
                Ok(Some(Arc::clone(unit)))
            },
            None => {
                warn!("No suitable unit found for task {} of type {:?}. Consider adjusting resource requirements or adding more processing units.", task.id, task.unit_type);
                Ok(None)
            },
        }
    }



    fn offload_to_cloud(&self, task: &Task) -> Result<(), XpuOptimizerError> {
        let cloud_offloader = self.cloud_offloader.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock cloud offloader: {}", e)))?;

        cloud_offloader.offload_task(task)
            .map_err(|e| {
                error!("Cloud offloading error for task {}: {}", task.id, e);
                XpuOptimizerError::CloudOffloadingError(format!("Failed to offload task {}: {}", task.id, e))
            })
    }

    fn execute_task_on_unit(&mut self, task: &Task, unit: Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>) -> Result<(u32, Duration), XpuOptimizerError> {
        let mut unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
        let unit_type = unit_guard.get_unit_type()?;

        if unit_type == task.unit_type {
            let result = unit_guard.execute_task(task);
            drop(unit_guard); // Release the lock before updating task history
            match result {
                Ok(duration) => {
                    self.update_task_history(task.id, duration, true)?;
                    Ok((task.id as u32, duration))
                },
                Err(e) => {
                    self.update_task_history(task.id, Duration::from_secs(0), false)?;
                    Err(XpuOptimizerError::TaskExecutionError(format!("Error executing task {} on {:?}: {}", task.id, unit_type, e)))
                }
            }
        } else {
            Err(XpuOptimizerError::ProcessingUnitNotFound(format!("Expected {:?}, found {:?}", task.unit_type, unit_type)))
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
            .ok_or_else(|| XpuOptimizerError::TaskNotFoundError(task_id))?;

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
        let mut current_unit_index = 0;

        self.scheduled_tasks.clear();
        for task in &self.task_queue {
            let mut assigned = false;
            for _ in 0..self.processing_units.len() {
                let unit = &self.processing_units[current_unit_index];
                if let Ok(can_handle) = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?.can_handle_task(task) {
                    if can_handle {
                        unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?.assign_task(task)?;
                        self.scheduled_tasks.push(task.clone());
                        current_unit_index = (current_unit_index + 1) % self.processing_units.len();
                        assigned = true;
                        break;
                    }
                }
                current_unit_index = (current_unit_index + 1) % self.processing_units.len();
            }
            if !assigned {
                warn!("No suitable processing unit found for task {}", task.id);
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
                .ok_or_else(|| {
                    XpuOptimizerError::SchedulingError("No suitable processing unit available".to_string())
                })?;

            let mut best_unit_guard = best_unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            best_unit_guard.assign_task(task)?;
            self.scheduled_tasks.push(task.clone());
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
                .ok_or_else(|| XpuOptimizerError::SchedulingError("No suitable processing unit available".to_string()))?;

            let mut best_unit_guard = best_unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock best unit: {}", e)))?;
            best_unit_guard.assign_task(task)?;
            self.scheduled_tasks.push(task.clone());
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
            .ok_or_else(|| XpuOptimizerError::TaskNotFoundError(prediction.task_id))?;

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

    // Removed unused functions: integrate_cuda, integrate_tensorflow, integrate_pytorch, connect_slurm, and connect_kubernetes

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

    // Removed unused function connect_mesos

    pub fn authenticate_user(
        &mut self,
        username: &str,
        password: &str,
    ) -> Result<String, XpuOptimizerError> {
        let user = self.users
            .get(username)
            .ok_or_else(|| XpuOptimizerError::UserNotFoundError(username.to_string()))?;

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

    fn remove_user(&mut self, username: &str) -> Result<(), XpuOptimizerError> {
        if self.users.remove(username).is_none() {
            return Err(XpuOptimizerError::UserNotFoundError(username.to_string()));
        }
        // Remove any active sessions for the removed user
        self.sessions.retain(|_, session| session.user_id != username);
        log::info!("User {} has been removed", username);
        Ok(())
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

    fn check_user_permission(
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

    fn validate_jwt_token(&self, token: &str) -> Result<String, XpuOptimizerError> {
        let decoding_key = DecodingKey::from_secret(self.jwt_secret.as_ref());
        let validation = Validation::new(Algorithm::HS256);
        let token_data = decode::<Claims>(token, &decoding_key, &validation)
            .map_err(|e| XpuOptimizerError::AuthenticationError(e.to_string()))?;
        Ok(token_data.claims.sub)
    }

    fn get_user_from_token(&self, token: &str) -> Result<User, XpuOptimizerError> {
        let username = self.validate_jwt_token(token)?;

        // Check if the token has expired
        if let Some(session) = self.sessions.get(token) {
            if Utc::now() > session.expiration {
                return Err(XpuOptimizerError::InvalidSessionError);
            }
        } else {
            return Err(XpuOptimizerError::SessionNotFoundError);
        }

        self.users
            .get(&username)
            .cloned()
            .ok_or_else(|| XpuOptimizerError::UserNotFoundError(username))
    }

    fn create_session(&mut self, username: &str) -> Result<String, XpuOptimizerError> {
        let user = self.users
            .get(username)
            .ok_or_else(|| XpuOptimizerError::UserNotFoundError(username.to_string()))?;
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

    pub fn get_memory_usage_timeline(&self) -> Result<Vec<MemoryUsagePoint>, XpuOptimizerError> {
        let memory_manager = self.memory_manager.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock memory manager: {}", e)))?;

        let total_memory = self.config.memory_pool_size;
        let available_memory = memory_manager.get_available_memory();
        let used_memory = total_memory - available_memory;

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| XpuOptimizerError::SystemTimeError(e.to_string()))?
            .as_secs();

        Ok(vec![MemoryUsagePoint {
            timestamp: current_time,
            used_memory,
            total_memory,
        }])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryUsagePoint {
    pub timestamp: u64,
    pub used_memory: usize,
    pub total_memory: usize,
}

impl PartialOrd for MemoryUsagePoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MemoryUsagePoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

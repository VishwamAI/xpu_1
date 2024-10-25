use crate::power_management::{EnergyProfile, PowerState, PowerManagementError};
use crate::XpuOptimizerError;
use crate::ml_models::MLModel;
use crate::task_data::{HistoricalTaskData, TaskExecutionData};
use crate::TaskPrediction;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum ProcessingUnitType {
    CPU,
    GPU,
    TPU,
    LPU,
    NPU,
    FPGA,
    VPU,
}

#[derive(Debug, Clone)]
pub struct DefaultMLModel;

impl MLModel for DefaultMLModel {
    fn predict(&self, _data: &HistoricalTaskData) -> Result<TaskPrediction, XpuOptimizerError> {
        Ok(TaskPrediction {
            task_id: 0,
            estimated_duration: Duration::from_secs(1),
            estimated_resource_usage: 1024,
            recommended_processing_unit: ProcessingUnitType::CPU,
        })
    }

    fn train(&mut self, _historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError> {
        Ok(()) // Default implementation does no training
    }

    fn clone_box(&self) -> Arc<Mutex<dyn MLModel + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn set_policy(&mut self, _policy: &str) -> Result<(), XpuOptimizerError> {
        Ok(()) // Default implementation ignores policy changes
    }
}

impl Default for DefaultMLModel {
    fn default() -> Self {
        DefaultMLModel
    }
}

impl Ord for ProcessingUnitType {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_string().cmp(&other.to_string())
    }
}

impl PartialOrd for ProcessingUnitType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SchedulerType {
    RoundRobin,
    LoadBalancing,
    AIPredictive,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MemoryManagerType {
    Simple,
    Dynamic,
}

impl fmt::Display for ProcessingUnitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait ProcessingUnitTrait: Send + Sync + Debug {
    fn get_id(&self) -> usize;
    fn get_unit_type(&self) -> Result<ProcessingUnitType, XpuOptimizerError>;
    fn get_current_load(&self) -> Result<Duration, XpuOptimizerError>;
    fn get_processing_power(&self) -> Result<f64, XpuOptimizerError>;
    fn get_power_state(&self) -> Result<PowerState, XpuOptimizerError>;
    fn get_energy_profile(&self) -> Result<&EnergyProfile, XpuOptimizerError>;
    fn get_load_percentage(&self) -> Result<f64, XpuOptimizerError>;
    fn can_handle_task(&self, task: &Task) -> Result<bool, XpuOptimizerError>;
    fn assign_task(&mut self, task: &Task) -> Result<(), XpuOptimizerError>;
    fn set_power_state(&mut self, state: PowerState) -> Result<(), PowerManagementError>;
    fn get_available_capacity(&self) -> Result<Duration, XpuOptimizerError>;
    fn execute_task(&mut self, task: &Task) -> Result<Duration, XpuOptimizerError>;
    fn clone_box(&self) -> Box<dyn ProcessingUnitTrait + Send + Sync>;
    fn set_energy_profile(&mut self, profile: EnergyProfile) -> Result<(), PowerManagementError>;
    fn as_any(&self) -> &dyn std::any::Any;
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessingUnit {
    pub id: usize,
    pub unit_type: ProcessingUnitType,
    pub current_load: Duration,
    pub processing_power: f64,
    pub power_state: PowerState,
    pub energy_profile: EnergyProfile,
}

impl ProcessingUnitTrait for ProcessingUnit {
    fn get_id(&self) -> usize {
        self.id
    }

    fn get_unit_type(&self) -> Result<ProcessingUnitType, XpuOptimizerError> {
        Ok(self.unit_type.clone())
    }

    fn get_current_load(&self) -> Result<Duration, XpuOptimizerError> {
        Ok(self.current_load)
    }

    fn get_processing_power(&self) -> Result<f64, XpuOptimizerError> {
        Ok(self.processing_power)
    }

    fn get_power_state(&self) -> Result<PowerState, XpuOptimizerError> {
        Ok(self.power_state.clone())
    }

    fn get_energy_profile(&self) -> Result<&EnergyProfile, XpuOptimizerError> {
        Ok(&self.energy_profile)
    }

    fn get_load_percentage(&self) -> Result<f64, XpuOptimizerError> {
        if self.processing_power == 0.0 {
            return Err(XpuOptimizerError::DivisionByZeroError("Processing power is zero".to_string()));
        }
        Ok(self.current_load.as_secs_f64() / self.processing_power)
    }

    fn can_handle_task(&self, task: &Task) -> Result<bool, XpuOptimizerError> {
        Ok(self.unit_type == task.unit_type &&
           self.current_load.saturating_add(task.execution_time) <= Duration::from_secs_f64(self.processing_power))
    }

    fn assign_task(&mut self, task: &Task) -> Result<(), XpuOptimizerError> {
        if self.can_handle_task(task)? {
            self.current_load = self.current_load.saturating_add(task.execution_time);
            Ok(())
        } else {
            Err(XpuOptimizerError::ResourceAllocationError(
                format!("Insufficient capacity on {} for task {}", self.unit_type, task.id)
            ))
        }
    }

    fn set_power_state(&mut self, state: PowerState) -> Result<(), PowerManagementError> {
        self.power_state = state;
        Ok(())
    }

    fn get_available_capacity(&self) -> Result<Duration, XpuOptimizerError> {
        Ok(Duration::from_secs_f64(self.processing_power).saturating_sub(self.current_load))
    }

    fn execute_task(&mut self, task: &Task) -> Result<Duration, XpuOptimizerError> {
        self.assign_task(task)?;
        let execution_time = task.execution_time;
        self.current_load = self.current_load.saturating_add(execution_time);
        Ok(execution_time)
    }

    fn set_energy_profile(&mut self, profile: EnergyProfile) -> Result<(), PowerManagementError> {
        self.energy_profile = profile;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn ProcessingUnitTrait + Send + Sync> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Removed duplicate Clone implementation for ProcessingUnit

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Task {
    pub id: usize,
    pub priority: u8,
    pub dependencies: Vec<usize>,
    pub execution_time: Duration,
    pub memory_requirement: usize,
    pub secure: bool,
    pub estimated_duration: Duration,
    pub estimated_resource_usage: usize,
    pub unit_type: ProcessingUnitType,
}

impl Task {
    pub fn new(
        id: usize,
        priority: u8,
        dependencies: Vec<usize>,
        execution_time: Duration,
        memory_requirement: usize,
        secure: bool,
        unit_type: ProcessingUnitType,
    ) -> Self {
        Task {
            id,
            priority,
            dependencies,
            execution_time,
            memory_requirement,
            secure,
            estimated_duration: Duration::default(),
            estimated_resource_usage: 0,
            unit_type,
        }
    }

    pub fn new_estimated(
        id: usize,
        priority: u8,
        dependencies: Vec<usize>,
        estimated_duration: Duration,
        estimated_resource_usage: usize,
        unit_type: ProcessingUnitType,
    ) -> Self {
        Task {
            id,
            priority,
            dependencies,
            execution_time: Duration::default(),
            memory_requirement: estimated_resource_usage,
            secure: false,
            estimated_duration,
            estimated_resource_usage,
            unit_type,
        }
    }
}

pub struct OptimizationMetrics {
    pub total_duration: Duration,
    pub average_latency: Duration,
    pub average_load: f32,
}

impl Default for OptimizationMetrics {
    fn default() -> Self {
        OptimizationMetrics {
            total_duration: Duration::default(),
            average_latency: Duration::default(),
            average_load: 0.0,
        }
    }
}

pub trait TaskScheduler: Send + Sync {
    fn schedule(&self, tasks: &[Task], units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>, XpuOptimizerError>;
    fn clone_box(&self) -> Arc<Mutex<dyn TaskScheduler + Send + Sync>>;
    fn generate_token(&self) -> Result<String, XpuOptimizerError>;
}

#[derive(Debug, Clone)]
pub enum Scheduler {
    RoundRobin(RoundRobinScheduler),
    LoadBalancing(LoadBalancingScheduler),
    AIOptimized(AIOptimizedScheduler),
}

impl Scheduler {
    pub fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }
}

impl TaskScheduler for Scheduler {
    fn schedule(&self, tasks: &[Task], units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>, XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.schedule(tasks, units),
            Scheduler::LoadBalancing(s) => s.schedule(tasks, units),
            Scheduler::AIOptimized(s) => s.schedule(tasks, units),
        }
    }

    fn clone_box(&self) -> Arc<Mutex<dyn TaskScheduler + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }
}

impl Scheduler {
    pub fn lock(&self) -> Result<Box<dyn TaskScheduler + Send + Sync>, XpuOptimizerError> {
        Ok(Box::new(self.clone()))
    }
}

impl Scheduler {
    pub fn new(scheduler_type: SchedulerType, ml_model: Option<Arc<Mutex<dyn MLModel + Send + Sync>>>) -> Self {
        match scheduler_type {
            SchedulerType::RoundRobin => Scheduler::RoundRobin(RoundRobinScheduler::new()),
            SchedulerType::LoadBalancing => Scheduler::LoadBalancing(LoadBalancingScheduler::new()),
            SchedulerType::AIPredictive => {
                if let Some(model) = ml_model {
                    Scheduler::AIOptimized(AIOptimizedScheduler::new(model))
                } else {
                    // Fallback to RoundRobin if no ML model is provided
                    Scheduler::RoundRobin(RoundRobinScheduler::new())
                }
            }
        }
    }
}

// The Clone implementation for Scheduler is already provided above.
// Removing the duplicate implementation to avoid conflicts.

impl std::fmt::Display for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scheduler::RoundRobin(_) => write!(f, "RoundRobin"),
            Scheduler::LoadBalancing(_) => write!(f, "LoadBalancing"),
            Scheduler::AIOptimized(_) => write!(f, "AIOptimized"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoundRobinScheduler;

impl RoundRobinScheduler {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for RoundRobinScheduler {
    fn default() -> Self {
        RoundRobinScheduler
    }
}

impl TaskScheduler for RoundRobinScheduler {
    fn schedule(&self, tasks: &[Task], units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>, XpuOptimizerError> {
        if units.is_empty() {
            return Err(XpuOptimizerError::SchedulingError("No processing units available".to_string()));
        }

        let mut scheduled_tasks = Vec::new();
        let mut unit_index = 0;

        for task in tasks {
            let mut task_scheduled = false;
            for _ in 0..units.len() {
                let unit = &units[unit_index];
                let can_handle = unit.lock()
                    .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?
                    .can_handle_task(task)?;

                if can_handle {
                    let mut unit_guard = unit.lock()
                        .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?;
                    unit_guard.assign_task(task)?;
                    drop(unit_guard); // Explicitly drop the lock before cloning
                    scheduled_tasks.push((task.clone(), Arc::clone(unit)));
                    task_scheduled = true;
                    break;
                }
                unit_index = (unit_index + 1) % units.len();
            }
            if !task_scheduled {
                return Err(XpuOptimizerError::SchedulingError(format!("No suitable processing unit for task {}", task.id)));
            }
            unit_index = (unit_index + 1) % units.len();
        }

        Ok(scheduled_tasks)
    }

    fn clone_box(&self) -> Arc<Mutex<dyn TaskScheduler + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }
}

#[derive(Debug, Clone)]
pub struct LoadBalancingScheduler;

impl LoadBalancingScheduler {
    pub fn new() -> Self {
        LoadBalancingScheduler
    }
}

impl Default for LoadBalancingScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskScheduler for LoadBalancingScheduler {
    fn schedule(&self, tasks: &[Task], units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>, XpuOptimizerError> {
        let mut scheduled_tasks = Vec::new();
        let mut unit_loads: Vec<Duration> = Vec::with_capacity(units.len());

        for unit in units {
            let load = unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?
                .get_current_load()?;
            unit_loads.push(load);
        }

        for task in tasks {
            let (assigned_unit_index, assigned_unit) = units.iter().enumerate()
                .filter_map(|(index, unit)| {
                    let guard = unit.lock().ok()?;
                    guard.can_handle_task(task).ok().and_then(|can_handle| {
                        if can_handle {
                            Some((index, Arc::clone(unit)))
                        } else {
                            None
                        }
                    })
                })
                .min_by_key(|(index, _)| unit_loads[*index])
                .ok_or_else(|| XpuOptimizerError::SchedulingError(format!("No suitable processing unit for task {}", task.id)))?;

            let mut unit_guard = assigned_unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?;

            unit_guard.assign_task(task)?;
            scheduled_tasks.push((task.clone(), assigned_unit.clone()));
            unit_loads[assigned_unit_index] = unit_loads[assigned_unit_index].saturating_add(task.execution_time);
        }

        Ok(scheduled_tasks)
    }

    fn clone_box(&self) -> Arc<Mutex<dyn TaskScheduler + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }
}

#[derive(Debug, Clone)]
pub struct AIOptimizedScheduler {
    ml_model: Arc<Mutex<dyn MLModel + Send + Sync>>,
}

impl AIOptimizedScheduler {
    pub fn new(ml_model: Arc<Mutex<dyn MLModel + Send + Sync>>) -> Self {
        AIOptimizedScheduler { ml_model }
    }
}

impl Default for AIOptimizedScheduler {
    fn default() -> Self {
        Self::new(Arc::new(Mutex::new(DefaultMLModel::default())))
    }
}

impl TaskScheduler for AIOptimizedScheduler {
    fn schedule(&self, tasks: &[Task], units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>, XpuOptimizerError> {
        let mut scheduled_tasks = Vec::new();

        for task in tasks {
            let historical_data = HistoricalTaskData {
                task_id: task.id,
                execution_time: task.execution_time,
                memory_usage: task.memory_requirement,
                unit_type: task.unit_type.clone(),
                priority: task.priority,
            };

            let prediction = self.ml_model.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?
                .predict(&historical_data)
                .map_err(|e| XpuOptimizerError::MLOptimizationError(format!("Failed to predict task execution: {}", e)))?;

            log::debug!("ML model prediction for task {}: {:?}", task.id, prediction);

            let best_unit = units
                .iter()
                .filter_map(|unit| {
                    let unit_guard = unit.lock()
                        .map_err(|e| {
                            log::warn!("Failed to lock processing unit: {}", e);
                            XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e))
                        })
                        .ok()?;
                    unit_guard.can_handle_task(task)
                        .map_err(|e| {
                            log::warn!("Error checking if unit can handle task: {}", e);
                            e
                        })
                        .ok()
                        .and_then(|can_handle| if can_handle { Some(Arc::clone(unit)) } else { None })
                })
                .min_by_key(|unit| {
                    unit.lock()
                        .map_err(|e| {
                            log::warn!("Failed to lock processing unit for load check: {}", e);
                            XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e))
                        })
                        .and_then(|guard| guard.get_current_load())
                        .map(|load| load.saturating_add(prediction.estimated_duration))
                        .unwrap_or(Duration::MAX)
                })
                .ok_or_else(|| {
                    log::error!("No suitable processing units available for task {}", task.id);
                    XpuOptimizerError::SchedulingError(format!("No suitable processing units available for task {}", task.id))
                })?;

            best_unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock best unit: {}", e)))?
                .assign_task(task)
                .map_err(|e| {
                    log::error!("Failed to assign task {} to best unit: {}", task.id, e);
                    XpuOptimizerError::TaskExecutionError(format!("Failed to assign task: {}", e))
                })?;

            log::info!("Scheduled task {} on best unit", task.id);
            scheduled_tasks.push((task.clone(), Arc::clone(&best_unit)));
        }

        Ok(scheduled_tasks)
    }

    fn clone_box(&self) -> Arc<Mutex<dyn TaskScheduler + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }
}

pub fn calculate_optimization_metrics(
    completed_tasks: &[(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)],
    start_time: Instant,
) -> Result<OptimizationMetrics, XpuOptimizerError> {
    let total_duration = start_time.elapsed();
    let total_latency: Duration = completed_tasks
        .iter()
        .map(|(task, _)| task.execution_time)
        .sum();
    let average_latency = if !completed_tasks.is_empty() {
        total_latency / completed_tasks.len() as u32
    } else {
        Duration::new(0, 0)
    };
    let average_load = completed_tasks
        .iter()
        .try_fold(0.0, |acc, (_, unit)| {
            let guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            Ok::<f64, XpuOptimizerError>(acc + guard.get_load_percentage().unwrap_or(0.0))
        })?;
    let average_load = if !completed_tasks.is_empty() {
        average_load / completed_tasks.len() as f64
    } else {
        0.0
    };

    Ok(OptimizationMetrics {
        total_duration,
        average_latency,
        average_load: average_load as f32,
    })
}

pub fn apply_optimization(
    tasks: &mut [Task],
    units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>],
    params: &OptimizationParams,
) -> Result<(), XpuOptimizerError> {
    for task in tasks.iter_mut() {
        task.priority = ((task.priority as f64) * params.task_priority_weight) as u8;
    }

    for unit in units.iter() {
        let mut unit_guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
        let mut energy_profile = unit_guard.get_energy_profile()?.clone();
        energy_profile.consumption_rate *= params.power_efficiency_factor;
        unit_guard.set_energy_profile(energy_profile)?;
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParams {
    pub load_balance_threshold: f64,
    pub prediction_weight: f64,
    pub task_priority_weight: f64,
    pub power_efficiency_factor: f64,
}

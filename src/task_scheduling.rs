use crate::power_management::{EnergyProfile, PowerState, PowerManagementError};
use crate::XpuOptimizerError;
use crate::ml_models::MLModel;
use crate::task_data::{HistoricalTaskData, TaskPrediction, TaskExecutionData};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;
use std::collections::VecDeque;

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
    AIOptimized,
    MLDriven,
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

pub struct PerformanceMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub resource_utilization: f64,
}

pub trait ProcessingUnitTrait: Send + Sync + Debug {
    fn get_id(&self) -> usize;
    fn get_unit_type(&self) -> Result<ProcessingUnitType, XpuOptimizerError>;
    fn get_current_load(&self) -> Result<Duration, XpuOptimizerError>;
    fn get_processing_power(&self) -> Result<f64, XpuOptimizerError>;
    fn get_power_state(&self) -> Result<PowerState, XpuOptimizerError>;
    fn get_energy_profile(&self) -> Result<&EnergyProfile, XpuOptimizerError>;
    fn get_load_percentage(&self) -> Result<f64, XpuOptimizerError>;
    fn get_unit_type_match(&self, task_unit_type: &ProcessingUnitType) -> Result<bool, XpuOptimizerError>;
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
    pub unit_type_match: Vec<ProcessingUnitType>,
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

    fn get_unit_type_match(&self, task_unit_type: &ProcessingUnitType) -> Result<bool, XpuOptimizerError> {
        Ok(self.unit_type == *task_unit_type)
    }

    fn can_handle_task(&self, task: &Task) -> Result<bool, XpuOptimizerError> {
        let unit_type_match = self.get_unit_type_match(&task.unit_type)?;
        let has_capacity = self.get_available_capacity()? >= task.execution_time;
        Ok(unit_type_match && has_capacity)
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
    fn generate_token(&self) -> Result<String, XpuOptimizerError>;
    fn find_best_unit(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError>;
    fn fallback_schedule(&self, task: &Task, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError>;
    fn find_best_unit_ml_driven(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError>;
    fn set_ml_driven_mode(&mut self, enabled: bool) -> Result<(), XpuOptimizerError>;
    fn update_ml_model(&mut self, historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError>;
    fn get_ml_predictions(&self, tasks: &[Task]) -> Result<Vec<TaskPrediction>, XpuOptimizerError>;
    fn optimize_ml_model(&mut self) -> Result<(), XpuOptimizerError>;
    fn adjust_scheduling_parameters(&mut self, performance_metrics: &PerformanceMetrics) -> Result<(), XpuOptimizerError>;
}

#[derive(Debug, Clone)]
pub enum Scheduler {
    RoundRobin(RoundRobinScheduler),
    LoadBalancing(LoadBalancingScheduler),
    AIOptimized(AIOptimizedScheduler),
}

impl Scheduler {
    pub fn set_ml_driven_mode(&mut self, enabled: bool) -> Result<(), XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.set_ml_driven_mode(enabled),
            Scheduler::LoadBalancing(s) => s.set_ml_driven_mode(enabled),
            Scheduler::AIOptimized(s) => s.set_ml_driven_mode(enabled),
        }
    }
}

impl Scheduler {
    pub fn schedule(&self, tasks: &[Task], units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Vec<(Task, Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>)>, XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.schedule(tasks, units),
            Scheduler::LoadBalancing(s) => s.schedule(tasks, units),
            Scheduler::AIOptimized(s) => s.schedule(tasks, units),
        }
    }
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

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }

    fn find_best_unit(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.find_best_unit(task, prediction, units),
            Scheduler::LoadBalancing(s) => s.find_best_unit(task, prediction, units),
            Scheduler::AIOptimized(s) => s.find_best_unit(task, prediction, units),
        }
    }

    fn fallback_schedule(&self, task: &Task, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.fallback_schedule(task, units),
            Scheduler::LoadBalancing(s) => s.fallback_schedule(task, units),
            Scheduler::AIOptimized(s) => s.fallback_schedule(task, units),
        }
    }

    fn find_best_unit_ml_driven(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.find_best_unit_ml_driven(task, prediction, units),
            Scheduler::LoadBalancing(s) => s.find_best_unit_ml_driven(task, prediction, units),
            Scheduler::AIOptimized(s) => s.find_best_unit_ml_driven(task, prediction, units),
        }
    }

    fn set_ml_driven_mode(&mut self, enabled: bool) -> Result<(), XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.set_ml_driven_mode(enabled),
            Scheduler::LoadBalancing(s) => s.set_ml_driven_mode(enabled),
            Scheduler::AIOptimized(s) => s.set_ml_driven_mode(enabled),
        }
    }

    fn update_ml_model(&mut self, historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.update_ml_model(historical_data),
            Scheduler::LoadBalancing(s) => s.update_ml_model(historical_data),
            Scheduler::AIOptimized(s) => s.update_ml_model(historical_data),
        }
    }

    fn get_ml_predictions(&self, tasks: &[Task]) -> Result<Vec<TaskPrediction>, XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.get_ml_predictions(tasks),
            Scheduler::LoadBalancing(s) => s.get_ml_predictions(tasks),
            Scheduler::AIOptimized(s) => s.get_ml_predictions(tasks),
        }
    }

    fn optimize_ml_model(&mut self) -> Result<(), XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.optimize_ml_model(),
            Scheduler::LoadBalancing(s) => s.optimize_ml_model(),
            Scheduler::AIOptimized(s) => s.optimize_ml_model(),
        }
    }

    fn adjust_scheduling_parameters(&mut self, performance_metrics: &PerformanceMetrics) -> Result<(), XpuOptimizerError> {
        match self {
            Scheduler::RoundRobin(s) => s.adjust_scheduling_parameters(performance_metrics),
            Scheduler::LoadBalancing(s) => s.adjust_scheduling_parameters(performance_metrics),
            Scheduler::AIOptimized(s) => s.adjust_scheduling_parameters(performance_metrics),
        }
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
            SchedulerType::RoundRobin => {
                log::info!("Creating RoundRobin scheduler");
                Scheduler::RoundRobin(RoundRobinScheduler::new())
            },
            SchedulerType::LoadBalancing => {
                log::info!("Creating LoadBalancing scheduler");
                Scheduler::LoadBalancing(LoadBalancingScheduler::new())
            },
            SchedulerType::AIPredictive | SchedulerType::AIOptimized | SchedulerType::MLDriven => {
                if let Some(model) = ml_model {
                    match scheduler_type {
                        SchedulerType::MLDriven => {
                            log::info!("Creating MLDriven scheduler with provided ML model");
                            Scheduler::AIOptimized(AIOptimizedScheduler::new_ml_driven(model))
                        },
                        SchedulerType::AIPredictive | SchedulerType::AIOptimized => {
                            log::info!("Creating AI-based scheduler ({:?}) with provided ML model", scheduler_type);
                            Scheduler::AIOptimized(AIOptimizedScheduler::new(model))
                        },
                        _ => unreachable!(),
                    }
                } else {
                    log::warn!("No ML model provided for {:?} scheduling. Falling back to RoundRobin.", scheduler_type);
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
pub struct RoundRobinScheduler {
    ml_driven: bool,
    current_index: usize,
}

impl RoundRobinScheduler {
    pub fn new() -> Self {
        RoundRobinScheduler {
            ml_driven: false,
            current_index: 0,
        }
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

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }

    fn find_best_unit(&self, task: &Task, _prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        for unit in units {
            let unit_guard = unit.lock()
                .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock processing unit: {}", e)))?;
            if unit_guard.can_handle_task(task)? {
                return Ok(Arc::clone(unit));
            }
        }
        Err(XpuOptimizerError::SchedulingError(format!("No suitable processing unit for task {}", task.id)))
    }

    fn fallback_schedule(&self, task: &Task, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        // For RoundRobinScheduler, fallback is the same as find_best_unit
        self.find_best_unit(task, &TaskPrediction::default(), units)
    }

    fn find_best_unit_ml_driven(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        // For RoundRobinScheduler, ML-driven approach is the same as regular find_best_unit
        self.find_best_unit(task, prediction, units)
    }

    fn set_ml_driven_mode(&mut self, enabled: bool) -> Result<(), XpuOptimizerError> {
        self.ml_driven = enabled;
        log::info!("ML-driven mode for RoundRobinScheduler set to: {}", enabled);
        Ok(())
    }

    fn update_ml_model(&mut self, _historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError> {
        // RoundRobinScheduler doesn't use ML model, so this is a no-op
        Ok(())
    }

    fn get_ml_predictions(&self, _tasks: &[Task]) -> Result<Vec<TaskPrediction>, XpuOptimizerError> {
        // RoundRobinScheduler doesn't use ML predictions, return empty vector
        Ok(Vec::new())
    }

    fn optimize_ml_model(&mut self) -> Result<(), XpuOptimizerError> {
        // RoundRobinScheduler doesn't use ML model, so this is a no-op
        Ok(())
    }

    fn adjust_scheduling_parameters(&mut self, _performance_metrics: &PerformanceMetrics) -> Result<(), XpuOptimizerError> {
        // RoundRobinScheduler doesn't adjust parameters, so this is a no-op
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LoadBalancingScheduler {
    current_index: usize,
    ml_driven: bool,
}

impl LoadBalancingScheduler {
    pub fn new() -> Self {
        LoadBalancingScheduler {
            current_index: 0,
            ml_driven: false,
        }
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

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }

    fn find_best_unit(&self, task: &Task, _prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        units.iter()
            .filter_map(|unit| {
                let guard = unit.lock().ok()?;
                if guard.can_handle_task(task).ok()? {
                    Some(Arc::clone(unit))
                } else {
                    None
                }
            })
            .min_by_key(|unit| {
                unit.lock().ok()
                    .and_then(|guard| guard.get_current_load().ok())
                    .unwrap_or(Duration::MAX)
            })
            .ok_or_else(|| XpuOptimizerError::SchedulingError(format!("No suitable processing unit for task {}", task.id)))
    }

    fn fallback_schedule(&self, task: &Task, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        self.find_best_unit(task, &TaskPrediction::default(), units)
    }

    fn find_best_unit_ml_driven(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        units.iter()
            .filter_map(|unit| {
                let guard = unit.lock().ok()?;
                if guard.can_handle_task(task).ok()? {
                    Some((Arc::clone(unit), guard.get_unit_type().ok()?))
                } else {
                    None
                }
            })
            .min_by(|(unit1, type1), (unit2, type2)| {
                let load1 = unit1.lock().ok().and_then(|guard| guard.get_current_load().ok()).unwrap_or(Duration::MAX);
                let load2 = unit2.lock().ok().and_then(|guard| guard.get_current_load().ok()).unwrap_or(Duration::MAX);

                if type1 == &prediction.recommended_processing_unit && type2 != &prediction.recommended_processing_unit {
                    std::cmp::Ordering::Less
                } else if type1 != &prediction.recommended_processing_unit && type2 == &prediction.recommended_processing_unit {
                    std::cmp::Ordering::Greater
                } else {
                    load1.cmp(&load2)
                }
            })
            .map(|(unit, _)| unit)
            .ok_or_else(|| XpuOptimizerError::SchedulingError(format!("No suitable processing unit for task {}", task.id)))
    }

    fn set_ml_driven_mode(&mut self, enabled: bool) -> Result<(), XpuOptimizerError> {
        self.ml_driven = enabled;
        log::info!("ML-driven mode for LoadBalancingScheduler set to: {}", enabled);
        Ok(())
    }

    fn update_ml_model(&mut self, _historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError> {
        // LoadBalancingScheduler doesn't use an ML model, so this is a no-op
        Ok(())
    }

    fn get_ml_predictions(&self, _tasks: &[Task]) -> Result<Vec<TaskPrediction>, XpuOptimizerError> {
        // LoadBalancingScheduler doesn't use ML predictions, so return an empty vector
        Ok(Vec::new())
    }

    fn optimize_ml_model(&mut self) -> Result<(), XpuOptimizerError> {
        // LoadBalancingScheduler doesn't use an ML model, so this is a no-op
        Ok(())
    }

    fn adjust_scheduling_parameters(&mut self, _performance_metrics: &PerformanceMetrics) -> Result<(), XpuOptimizerError> {
        // LoadBalancingScheduler doesn't adjust parameters, so this is a no-op
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AIOptimizedScheduler {
    ml_model: Arc<Mutex<dyn MLModel + Send + Sync>>,
    fallback_scheduler: RoundRobinScheduler,
    ml_driven: bool,
    prediction_confidence_threshold: f64,
    adaptive_threshold: bool,
    last_prediction_accuracy: f64,
    early_stopping_patience: usize,
    early_stopping_min_delta: f64,
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
    feature_selection_enabled: bool,
    feature_selection_threshold: f64,
    continuous_learning_enabled: bool,
    model_update_frequency: usize,
    ensemble_models: Vec<Arc<Mutex<dyn MLModel + Send + Sync>>>,
    prediction_history: VecDeque<(TaskPrediction, Duration)>,
}

impl AIOptimizedScheduler {
    pub fn set_ml_driven_mode(&mut self, enabled: bool) -> Result<(), XpuOptimizerError> {
        self.ml_driven = enabled;
        log::info!("ML-driven mode for AIOptimizedScheduler set to: {}", enabled);
        if enabled {
            log::info!("Adjusting AIOptimizedScheduler parameters for ML-driven mode");
            self.adaptive_threshold = true;
            self.feature_selection_enabled = true;
            self.prediction_confidence_threshold = 0.7;
            self.learning_rate = 0.001;
            self.batch_size = 64;
            self.epochs = 200;
            self.continuous_learning_enabled = true;
        }
        Ok(())
    }

    pub fn new(ml_model: Arc<Mutex<dyn MLModel + Send + Sync>>) -> Self {
        AIOptimizedScheduler {
            ml_model,
            fallback_scheduler: RoundRobinScheduler::new(),
            ml_driven: false,
            prediction_confidence_threshold: 0.7,
            adaptive_threshold: false,
            last_prediction_accuracy: 1.0,
            early_stopping_patience: 10,
            early_stopping_min_delta: 1e-6,
            learning_rate: 0.01,
            batch_size: 32,
            epochs: 100,
            feature_selection_enabled: false,
            feature_selection_threshold: 0.05,
            continuous_learning_enabled: false,
            model_update_frequency: 100,
            ensemble_models: Vec::new(),
            prediction_history: VecDeque::new(),
        }
    }

    pub fn new_ml_driven(ml_model: Arc<Mutex<dyn MLModel + Send + Sync>>) -> Self {
        let mut scheduler = Self::new(ml_model);
        scheduler.set_ml_driven_mode(true).unwrap();
        scheduler
    }

    pub fn set_prediction_confidence_threshold(&mut self, threshold: f64) {
        self.prediction_confidence_threshold = threshold.clamp(0.0, 1.0);
    }

    pub fn set_adaptive_threshold(&mut self, adaptive: bool) {
        self.adaptive_threshold = adaptive;
    }

    pub fn update_prediction_accuracy(&mut self, accuracy: f64) {
        self.last_prediction_accuracy = accuracy;
        if self.adaptive_threshold {
            self.adjust_confidence_threshold();
        }
    }

    fn adjust_confidence_threshold(&mut self) {
        let adjustment_factor = 0.1;
        if self.last_prediction_accuracy < 0.5 {
            self.prediction_confidence_threshold += adjustment_factor;
        } else {
            self.prediction_confidence_threshold -= adjustment_factor;
        }
        self.prediction_confidence_threshold = self.prediction_confidence_threshold.clamp(0.5, 0.9);
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
                .map_err(|e| {
                    log::error!("Failed to lock ML model: {}", e);
                    XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e))
                })?
                .predict(&historical_data)
                .map_err(|e| {
                    log::error!("Failed to predict task execution for task {}: {}", task.id, e);
                    XpuOptimizerError::MLOptimizationError(format!("Failed to predict task execution: {}", e))
                })?;

            log::debug!("ML model prediction for task {}: {:?}", task.id, prediction);

            let assigned_unit = if self.ml_driven {
                self.find_best_unit_ml_driven(task, &prediction, units)?
            } else {
                match self.find_best_unit(task, &prediction, units) {
                    Ok(unit) => unit,
                    Err(e) => {
                        log::warn!("Failed to find best unit for task {}, using fallback scheduler: {}", task.id, e);
                        self.fallback_schedule(task, units)?
                    }
                }
            };

            match assigned_unit.lock() {
                Ok(mut unit_guard) => {
                    if let Err(e) = unit_guard.assign_task(task) {
                        log::error!("Failed to assign task {} to assigned unit: {}", task.id, e);
                        return Err(XpuOptimizerError::TaskExecutionError(format!("Failed to assign task {}: {}", task.id, e)));
                    }
                },
                Err(e) => {
                    log::error!("Failed to lock assigned unit for task {}: {}", task.id, e);
                    return Err(XpuOptimizerError::LockError(format!("Failed to lock assigned unit: {}", e)));
                }
            }

            log::info!("Scheduled task {} on assigned unit", task.id);
            scheduled_tasks.push((task.clone(), Arc::clone(&assigned_unit)));
        }

        Ok(scheduled_tasks)
    }

    fn find_best_unit(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        let best_unit = units
            .iter()
            .filter_map(|unit| {
                let unit_guard = match unit.lock() {
                    Ok(guard) => guard,
                    Err(e) => {
                        log::warn!("Failed to lock processing unit for task {}: {}", task.id, e);
                        return None;
                    }
                };

                match unit_guard.can_handle_task(task) {
                    Ok(can_handle) if can_handle => {
                        match unit_guard.get_current_load() {
                            Ok(load) => Some((Arc::clone(unit), load)),
                            Err(e) => {
                                log::warn!("Failed to get current load for unit when scheduling task {}: {}", task.id, e);
                                None
                            }
                        }
                    },
                    Ok(_) => None,
                    Err(e) => {
                        log::warn!("Error checking if unit can handle task {}: {}", task.id, e);
                        None
                    }
                }
            })
            .min_by_key(|(_, load)| load.saturating_add(prediction.estimated_duration));

        best_unit
            .map(|(unit, _)| unit)
            .ok_or_else(|| {
                log::warn!("No suitable processing unit found for task {}. Falling back to RoundRobin scheduling.", task.id);
                XpuOptimizerError::SchedulingError(format!("No suitable processing unit for task {}", task.id))
            })
    }

    fn find_best_unit_ml_driven(&self, task: &Task, prediction: &TaskPrediction, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        let best_unit = units.iter()
            .filter_map(|unit| {
                let unit_guard = match unit.lock() {
                    Ok(guard) => guard,
                    Err(e) => {
                        log::warn!("Failed to lock processing unit for task {}: {}", task.id, e);
                        return None;
                    }
                };

                match (unit_guard.can_handle_task(task), unit_guard.get_unit_type()) {
                    (Ok(true), Ok(unit_type)) => Some((Arc::clone(unit), unit_type, unit_guard.get_current_load().ok()?)),
                    _ => None
                }
            })
            .min_by(|(_, type1, load1), (_, type2, load2)| {
                let type_match1 = *type1 == prediction.recommended_processing_unit;
                let type_match2 = *type2 == prediction.recommended_processing_unit;
                match (type_match1, type_match2) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => load1.cmp(load2)
                }
            })
            .map(|(unit, _, _)| unit);

        best_unit.ok_or_else(|| {
            log::error!("No suitable processing unit found for task {}", task.id);
            XpuOptimizerError::SchedulingError(format!("No suitable processing unit for task {}", task.id))
        })
    }

    fn fallback_schedule(&self, task: &Task, units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>]) -> Result<Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>, XpuOptimizerError> {
        match self.fallback_scheduler.schedule(&[task.clone()], units)? {
            ref scheduled if !scheduled.is_empty() => Ok(Arc::clone(&scheduled[0].1)),
            _ => {
                log::error!("Fallback scheduler failed to assign task {}", task.id);
                Err(XpuOptimizerError::SchedulingError(format!("Fallback scheduler failed to assign task {}", task.id)))
            }
        }
    }

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }

    fn set_ml_driven_mode(&mut self, enabled: bool) -> Result<(), XpuOptimizerError> {
        self.set_ml_driven_mode(enabled)
    }

    fn update_ml_model(&mut self, historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError> {
        if self.continuous_learning_enabled {
            let mut ml_model = self.ml_model.lock().map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?;
            ml_model.train(historical_data)?;
        }
        Ok(())
    }

    fn get_ml_predictions(&self, tasks: &[Task]) -> Result<Vec<TaskPrediction>, XpuOptimizerError> {
        let ml_model = self.ml_model.lock().map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?;
        tasks.iter().map(|task| {
            let historical_data = HistoricalTaskData {
                task_id: task.id,
                execution_time: task.execution_time,
                memory_usage: task.memory_requirement,
                unit_type: task.unit_type.clone(),
                priority: task.priority,
            };
            ml_model.predict(&historical_data)
        }).collect()
    }

    fn optimize_ml_model(&mut self) -> Result<(), XpuOptimizerError> {
        // Implement optimization logic for the ML model
        Ok(())
    }

    fn adjust_scheduling_parameters(&mut self, _performance_metrics: &PerformanceMetrics) -> Result<(), XpuOptimizerError> {
        // Implement logic to adjust scheduling parameters based on performance metrics
        Ok(())
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
        .try_fold(0.0, |acc, (_, unit)| -> Result<f64, XpuOptimizerError> {
            let guard = unit.lock().map_err(|e| XpuOptimizerError::LockError(e.to_string()))?;
            guard.get_load_percentage().and_then(|load| Ok(acc + load))
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

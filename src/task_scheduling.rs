use crate::power_management::{EnergyProfile, PowerState};
use crate::XpuOptimizerError;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum ProcessingUnitType {
    CPU,
    GPU,
    LPU,
    NPU,
    FPGA,
    VPU,
}

impl fmt::Display for ProcessingUnitType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingUnit {
    pub id: usize,
    pub unit_type: ProcessingUnitType,
    pub current_load: Duration,
    pub processing_power: f32,
    pub power_state: PowerState,
    pub energy_profile: EnergyProfile,
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

pub struct TaskScheduler {
    pub tasks: VecDeque<Task>,
    processing_units: Vec<ProcessingUnit>,
    historical_data: HashMap<ProcessingUnitType, Vec<(Duration, Duration)>>,
    last_load_balance: Instant,
    load_balance_threshold: f32,
    prediction_weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    pub id: usize,
    pub priority: u8,
    pub execution_time: Duration,
    pub memory_requirement: usize,
    pub unit_type: ProcessingUnitType,
    pub unit: ProcessingUnit,
    pub dependencies: Vec<usize>,
    pub secure: bool,
    pub estimated_duration: Duration,
    pub estimated_resource_usage: usize,
}

impl TaskScheduler {
    pub fn new(num_processing_units: usize) -> Self {
        let mut rng = rand::thread_rng();
        TaskScheduler {
            tasks: VecDeque::new(),
            processing_units: (0..num_processing_units)
                .map(|id| ProcessingUnit {
                    id,
                    unit_type: match rng.gen_range(0..6) {
                        0 => ProcessingUnitType::CPU,
                        1 => ProcessingUnitType::GPU,
                        2 => ProcessingUnitType::LPU,
                        3 => ProcessingUnitType::NPU,
                        4 => ProcessingUnitType::FPGA,
                        _ => ProcessingUnitType::VPU,
                    },
                    current_load: Duration::new(0, 0),
                    processing_power: 1.0,
                    power_state: PowerState::Normal,
                    energy_profile: EnergyProfile::default(),
                })
                .collect(),
            historical_data: HashMap::new(),
            last_load_balance: Instant::now(),
            load_balance_threshold: 0.7,
            prediction_weight: 0.5,
        }
    }

    pub fn add_task(&mut self, task: Task) {
        let position = self.tasks.iter().position(|t| t.priority < task.priority);
        match position {
            Some(index) => self.tasks.insert(index, task),
            None => self.tasks.push_back(task),
        }
    }

    pub fn get_next_task(&mut self) -> Option<Task> {
        self.tasks.pop_front()
    }

    pub fn schedule(&mut self) -> Result<Vec<Task>, XpuOptimizerError> {
        let (completed_tasks, _) = self.schedule_with_metrics()?;
        Ok(completed_tasks)
    }

    pub fn schedule_with_metrics(
        &mut self,
    ) -> Result<(Vec<Task>, OptimizationMetrics), XpuOptimizerError> {
        println!("Scheduling tasks with adaptive optimization...");
        let mut completed_tasks = Vec::new();
        let mut unscheduled_tasks = VecDeque::new();
        let max_retries = self.processing_units.len() * 2;
        let mut retry_count = 0;
        let mut total_latency = Duration::new(0, 0);
        let start_time = Instant::now();

        while let Some(task) = self.get_next_task() {
            match self.find_optimal_unit_index(&task) {
                Some(unit_index) => {
                    let unit = &mut self.processing_units[unit_index];
                    println!(
                        "Executing task {} on processing unit {} ({})",
                        task.id, unit.id, unit.unit_type
                    );
                    let task_start_time = Instant::now();
                    unit.current_load += task.execution_time;
                    let mut executed_task = task.clone();
                    executed_task.unit = unit.clone(); // Assign the unit to the task
                    completed_tasks.push(executed_task);
                    retry_count = 0;

                    // Update historical data
                    let actual_duration = task_start_time.elapsed();
                    total_latency += actual_duration;
                    self.historical_data
                        .entry(unit.unit_type.clone())
                        .or_default()
                        .push((task.execution_time, actual_duration));

                    // Perform adaptive optimization
                    self.adapt_scheduling_parameters(&completed_tasks);
                }
                None => {
                    println!("No available processing unit for task {}", task.id);
                    unscheduled_tasks.push_back(task);
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(XpuOptimizerError::SchedulingError(
                            "Max retries reached. Unable to schedule remaining tasks.".to_string(),
                        ));
                    }
                }
            }

            if self.tasks.is_empty() && !unscheduled_tasks.is_empty() {
                self.tasks.append(&mut unscheduled_tasks);
            }

            // Perform load balancing every 10 tasks
            if completed_tasks.len() % 10 == 0 {
                self.load_balance();
            }
        }

        self.tasks.append(&mut unscheduled_tasks);
        let total_duration = start_time.elapsed();
        let average_latency = if !completed_tasks.is_empty() {
            total_latency.div_f32(completed_tasks.len() as f32)
        } else {
            Duration::new(0, 0)
        };
        let average_load = self
            .processing_units
            .iter()
            .map(|unit| unit.current_load)
            .sum::<Duration>()
            .div_f32(self.processing_units.len() as f32);

        let metrics = OptimizationMetrics {
            total_duration,
            average_latency,
            average_load: average_load.as_secs_f32() / total_duration.as_secs_f32(),
        };

        Ok((completed_tasks, metrics))
    }

    fn find_optimal_unit_index(&self, task: &Task) -> Option<usize> {
        self.processing_units
            .iter()
            .enumerate()
            .min_by_key(|(_, unit)| {
                let predicted_duration = self.predict_duration(task, &unit.unit_type);
                unit.current_load + predicted_duration
            })
            .map(|(index, _)| index)
    }

    fn predict_duration(&self, task: &Task, unit_type: &ProcessingUnitType) -> Duration {
        if let Some(data) = self.historical_data.get(unit_type) {
            let sum: Duration = data.iter().map(|(_, actual)| *actual).sum();
            let count = data.len() as u32;
            if count > 0 {
                let historical_prediction = sum / count;
                let weighted_prediction = (historical_prediction.as_secs_f32()
                    * self.prediction_weight
                    + task.execution_time.as_secs_f32() * (1.0 - self.prediction_weight))
                    as u64;
                Duration::from_secs(weighted_prediction)
            } else {
                task.execution_time
            }
        } else {
            task.execution_time
        }
    }

    fn load_balance(&mut self) {
        if self.last_load_balance.elapsed() < Duration::from_secs(60) {
            return;
        }

        println!("Performing load balancing...");
        let total_load: Duration = self
            .processing_units
            .iter()
            .map(|unit| unit.current_load)
            .sum();
        let average_load = total_load / self.processing_units.len() as u32;

        for unit in &mut self.processing_units {
            unit.current_load = unit.current_load.clamp(Duration::ZERO, average_load * 2);
        }

        self.last_load_balance = Instant::now();
    }

    pub fn apply_optimization(&mut self, params: OptimizationParams) {
        println!("Applying optimization parameters: {:?}", params);

        // Adjust load balancing threshold
        self.load_balance_threshold = params.load_balance_threshold;

        // Update prediction weight for task duration estimation
        self.prediction_weight = params.prediction_weight;

        // Modify task priority calculation
        for task in &mut self.tasks {
            task.priority = ((task.priority as f32) * params.task_priority_weight) as u8;
        }

        // Adjust power efficiency settings for processing units
        for unit in &mut self.processing_units {
            unit.energy_profile.consumption_rate *= params.power_efficiency_factor;
        }

        println!("Optimization parameters applied successfully");
    }

    fn adapt_scheduling_parameters(&mut self, completed_tasks: &[Task]) {
        println!(
            "Adapting scheduling parameters based on {} completed tasks",
            completed_tasks.len()
        );

        // Calculate average execution time for each processing unit type
        let mut avg_execution_times: HashMap<ProcessingUnitType, Duration> = HashMap::new();
        for task in completed_tasks {
            avg_execution_times
                .entry(task.unit.unit_type.clone())
                .and_modify(|time| *time += task.execution_time)
                .or_insert(task.execution_time);
        }

        // Adjust processing power based on average execution times
        for (unit_type, avg_time) in avg_execution_times.iter() {
            if let Some(unit) = self
                .processing_units
                .iter_mut()
                .find(|u| u.unit_type == *unit_type)
            {
                let adjustment_factor = 1.0 / avg_time.as_secs_f32().max(1.0);
                unit.processing_power =
                    (unit.processing_power * 0.9 + adjustment_factor * 0.1).clamp(0.1, 2.0);
            }
        }

        // Adjust task priorities based on waiting time
        for task in self.tasks.iter_mut() {
            task.priority = task.priority.saturating_add(1);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParams {
    pub load_balance_threshold: f32,
    pub prediction_weight: f32,
    pub task_priority_weight: f32,
    pub power_efficiency_factor: f32,
}

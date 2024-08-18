use std::time::Duration;
use crate::{ProcessingUnitType, ProcessingUnit};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskExecutionData {
    pub id: usize,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub processing_unit: ProcessingUnitType,
    pub priority: u8,
    pub success: bool,
    pub memory_requirement: usize,
    pub unit: ProcessingUnit,
}

#[derive(Clone, Debug)]
pub struct HistoricalTaskData {
    pub task_id: usize,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub processing_unit: ProcessingUnitType,
    pub priority: u8,
}

#[derive(Clone, Debug)]
pub struct TaskPrediction {
    pub task_id: usize,
    pub estimated_duration: Duration,
    pub estimated_resource_usage: usize,
    pub recommended_processing_unit: ProcessingUnitType,
}

pub trait TaskDataManager {
    fn add_execution_data(&mut self, data: TaskExecutionData);
    fn get_historical_data(&self) -> Vec<HistoricalTaskData>;
    fn clear_old_data(&mut self, threshold: Duration);
}

pub struct InMemoryTaskDataManager {
    execution_data: Vec<TaskExecutionData>,
}

impl InMemoryTaskDataManager {
    pub fn new() -> Self {
        InMemoryTaskDataManager {
            execution_data: Vec::new(),
        }
    }
}

impl TaskDataManager for InMemoryTaskDataManager {
    fn add_execution_data(&mut self, data: TaskExecutionData) {
        self.execution_data.push(data);
    }

    fn get_historical_data(&self) -> Vec<HistoricalTaskData> {
        self.execution_data
            .iter()
            .map(|data| HistoricalTaskData {
                task_id: data.id,
                execution_time: data.execution_time,
                memory_usage: data.memory_usage,
                processing_unit: data.processing_unit.clone(),
                priority: data.priority,
            })
            .collect()
    }

    fn clear_old_data(&mut self, threshold: Duration) {
        let now = std::time::Instant::now();
        self.execution_data.retain(|data| now.duration_since(std::time::Instant::now()) < threshold);
    }
}

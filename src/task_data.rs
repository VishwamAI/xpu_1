use crate::task_scheduling::ProcessingUnitType;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskExecutionData {
    pub id: usize,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub unit_type: ProcessingUnitType,
    pub priority: u8,
    pub success: bool,
    pub memory_requirement: usize,
}

#[derive(Clone, Debug)]
pub struct HistoricalTaskData {
    pub task_id: usize,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub unit_type: ProcessingUnitType,
    pub priority: u8,
}

#[derive(Clone, Debug)]
pub struct TaskPrediction {
    pub task_id: usize,
    pub estimated_duration: Duration,
    pub estimated_resource_usage: usize,
    pub recommended_processing_unit: ProcessingUnitType,
}

impl Default for TaskPrediction {
    fn default() -> Self {
        TaskPrediction {
            task_id: 0,
            estimated_duration: Duration::default(),
            estimated_resource_usage: 0,
            recommended_processing_unit: ProcessingUnitType::CPU,
        }
    }
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

impl Default for InMemoryTaskDataManager {
    fn default() -> Self {
        Self::new()
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
                unit_type: data.unit_type.clone(),
                priority: data.priority,
            })
            .collect()
    }

    fn clear_old_data(&mut self, threshold: Duration) {
        let now = std::time::Instant::now();
        self.execution_data.retain(|data| {
            now.duration_since(std::time::Instant::now() - data.execution_time) < threshold
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    fn create_test_task_execution_data(id: usize) -> TaskExecutionData {
        TaskExecutionData {
            id,
            execution_time: Duration::from_secs(1),
            memory_usage: 1024,
            unit_type: ProcessingUnitType::CPU,
            priority: 1,
            success: true,
            memory_requirement: 2048,
        }
    }

    #[test]
    fn test_add_execution_data() {
        let mut manager = InMemoryTaskDataManager::new();
        let data = create_test_task_execution_data(1);
        manager.add_execution_data(data.clone());
        assert_eq!(manager.execution_data.len(), 1);
        assert_eq!(manager.execution_data[0].id, data.id);
    }

    #[test]
    fn test_get_historical_data() {
        let mut manager = InMemoryTaskDataManager::new();
        manager.add_execution_data(create_test_task_execution_data(1));
        manager.add_execution_data(create_test_task_execution_data(2));
        let historical_data = manager.get_historical_data();
        assert_eq!(historical_data.len(), 2);
        assert_eq!(historical_data[0].task_id, 1);
        assert_eq!(historical_data[1].task_id, 2);
    }

    #[test]
    fn test_clear_old_data() {
        let mut manager = InMemoryTaskDataManager::new();
        let old_data = TaskExecutionData {
            execution_time: Duration::from_secs(10),
            ..create_test_task_execution_data(1)
        };
        let new_data = create_test_task_execution_data(2);
        manager.add_execution_data(old_data);
        manager.add_execution_data(new_data);

        // Clear data older than 5 seconds
        manager.clear_old_data(Duration::from_secs(5));
        assert_eq!(manager.execution_data.len(), 1);
        assert_eq!(manager.execution_data[0].id, 2);
    }
}

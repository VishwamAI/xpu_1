use std::time::Duration;
use crate::task_data::{TaskExecutionData, HistoricalTaskData, TaskPrediction};
use crate::ProcessingUnitType;

pub trait MLModel: Send + Sync {
    fn train(&mut self, historical_data: &[TaskExecutionData]);
    fn predict(&self, task_data: &HistoricalTaskData) -> TaskPrediction;
    fn clone_box(&self) -> Box<dyn MLModel>;
}

impl Clone for Box<dyn MLModel> {
    fn clone(&self) -> Box<dyn MLModel> {
        self.clone_box()
    }
}

pub struct SimpleRegressionModel {
    coefficients: Vec<f64>,
}

impl SimpleRegressionModel {
    pub fn new() -> Self {
        SimpleRegressionModel {
            coefficients: vec![0.0; 4], // Assuming 4 features: execution_time, memory_usage, processing_unit_type, priority
        }
    }

    fn normalize_processing_unit(&self, unit: &ProcessingUnitType) -> f64 {
        match unit {
            ProcessingUnitType::CPU => 0.0,
            ProcessingUnitType::GPU => 1.0,
            ProcessingUnitType::NPU => 2.0,
            ProcessingUnitType::FPGA => 3.0,
        }
    }
}

impl MLModel for SimpleRegressionModel {
    fn train(&mut self, historical_data: &[TaskExecutionData]) {
        let mut x = Vec::new();
        let mut y = Vec::new();

        for data in historical_data {
            x.push(vec![
                data.execution_time.as_secs_f64(),
                data.memory_usage as f64,
                self.normalize_processing_unit(&data.processing_unit),
                data.priority as f64,
            ]);
            y.push(data.execution_time.as_secs_f64());
        }

        // Simple gradient descent
        let learning_rate = 0.01;
        let iterations = 1000;
        let m = x.len();

        for _ in 0..iterations {
            let mut gradient = vec![0.0; self.coefficients.len()];
            for i in 0..m {
                let h = x[i].iter().zip(&self.coefficients).map(|(xi, ci)| xi * ci).sum::<f64>();
                let error = h - y[i];
                for j in 0..self.coefficients.len() {
                    gradient[j] += error * x[i][j] / m as f64;
                }
            }
            for j in 0..self.coefficients.len() {
                self.coefficients[j] -= learning_rate * gradient[j];
            }
        }
    }

    fn predict(&self, task_data: &HistoricalTaskData) -> TaskPrediction {
        let features = vec![
            task_data.execution_time.as_secs_f64(),
            task_data.memory_usage as f64,
            self.normalize_processing_unit(&task_data.processing_unit),
            task_data.priority as f64,
        ];

        let prediction: f64 = self.coefficients.iter().zip(features.iter())
            .map(|(coef, feat)| coef * feat)
            .sum();

        TaskPrediction {
            task_id: task_data.task_id,
            estimated_duration: Duration::from_secs_f64(prediction),
            estimated_resource_usage: (prediction * 100.0) as usize, // Dummy resource usage calculation
            recommended_processing_unit: task_data.processing_unit.clone(), // Use the same processing unit as input for now
        }
    }

    fn clone_box(&self) -> Box<dyn MLModel> {
        Box::new(self.clone())
    }
}

impl Clone for SimpleRegressionModel {
    fn clone(&self) -> Self {
        SimpleRegressionModel {
            coefficients: self.coefficients.clone(),
        }
    }
}

use crate::task_data::{HistoricalTaskData, TaskExecutionData, TaskPrediction};
use crate::ProcessingUnitType;
use std::time::Duration;

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
            ProcessingUnitType::LPU => 4.0,
            ProcessingUnitType::VPU => 5.0,
        }
    }
}

impl Default for SimpleRegressionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl MLModel for SimpleRegressionModel {
    fn train(&mut self, historical_data: &[TaskExecutionData]) {
        let x: Vec<Vec<f64>> = historical_data
            .iter()
            .map(|data| {
                vec![
                    data.execution_time.as_secs_f64(),
                    data.memory_usage as f64,
                    self.normalize_processing_unit(&data.processing_unit),
                    data.priority as f64,
                ]
            })
            .collect();

        let y: Vec<f64> = historical_data
            .iter()
            .map(|data| data.execution_time.as_secs_f64())
            .collect();

        // Simple gradient descent
        let learning_rate = 0.01;
        let iterations = 1000;
        let m = x.len();

        for _ in 0..iterations {
            let mut gradient = vec![0.0; self.coefficients.len()];
            for (i, xi) in x.iter().enumerate() {
                let h: f64 = xi
                    .iter()
                    .zip(&self.coefficients)
                    .map(|(xi, ci)| xi * ci)
                    .sum();
                let error = h - y[i];
                for (j, &xij) in xi.iter().enumerate() {
                    gradient[j] += error * xij / m as f64;
                }
            }
            for (coeff, grad) in self.coefficients.iter_mut().zip(gradient.iter()) {
                *coeff -= learning_rate * grad;
            }
        }
    }

    fn predict(&self, task_data: &HistoricalTaskData) -> TaskPrediction {
        let features = [
            task_data.execution_time.as_secs_f64(),
            task_data.memory_usage as f64,
            self.normalize_processing_unit(&task_data.processing_unit),
            task_data.priority as f64,
        ];

        let prediction: f64 = self
            .coefficients
            .iter()
            .zip(features.iter())
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

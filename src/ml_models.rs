use crate::task_data::{HistoricalTaskData, TaskExecutionData, TaskPrediction};
use crate::task_scheduling::{ProcessingUnitType, ProcessingUnitTrait, Scheduler, AIOptimizedScheduler};
use crate::XpuOptimizerError;
use crate::xpu_optimization::MachineLearningOptimizer;
use std::time::Duration;
use std::sync::{Arc, Mutex};

pub trait MLModel: Send + Sync {
    fn train(&mut self, historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError>;
    fn predict(&self, task_data: &HistoricalTaskData) -> Result<TaskPrediction, XpuOptimizerError>;
    fn clone_box(&self) -> Arc<Mutex<dyn MLModel + Send + Sync>>;
    fn set_policy(&mut self, policy: &str) -> Result<(), XpuOptimizerError>;
    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Debug for dyn MLModel + Send + Sync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MLModel")
    }
}

#[derive(Debug, Clone)]
pub struct SimpleRegressionModel {
    coefficients: Vec<f64>,
    mean_values: Vec<f64>,
    std_values: Vec<f64>,
    policy: String,
}

impl SimpleRegressionModel {
    pub fn new() -> Self {
        SimpleRegressionModel {
            // Initialize with reasonable defaults based on typical task characteristics
            coefficients: vec![0.1, 0.001, 0.05, 0.02], // [execution_time_factor, memory_factor, unit_type_factor, priority_factor]
            policy: "default".to_string(),
            mean_values: vec![0.0; 4], // Initialize means for feature normalization
            std_values: vec![1.0; 4], // Initialize standard deviations
        }
    }

    fn normalize_processing_unit(&self, unit: &ProcessingUnitType) -> f64 {
        match unit {
            ProcessingUnitType::CPU => 0.0,
            ProcessingUnitType::GPU => 1.0,
            ProcessingUnitType::TPU => 2.0,
            ProcessingUnitType::NPU => 3.0,
            ProcessingUnitType::FPGA => 4.0,
            ProcessingUnitType::LPU => 5.0,
            ProcessingUnitType::VPU => 6.0,
        }
    }
}

impl Default for SimpleRegressionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl MLModel for SimpleRegressionModel {
    fn train(&mut self, historical_data: &[TaskExecutionData]) -> Result<(), XpuOptimizerError> {
        if historical_data.is_empty() {
            return Err(XpuOptimizerError::MLOptimizationError("No historical data provided for training".to_string()));
        }

        // Calculate mean values for feature normalization
        let mut mean_execution_time = 0.0;
        let mut mean_memory_usage = 0.0;
        let mut mean_priority = 0.0;

        for data in historical_data {
            mean_execution_time += data.execution_time.as_secs_f64();
            mean_memory_usage += data.memory_usage as f64;
            mean_priority += data.priority as f64;
        }

        let n = historical_data.len() as f64;
        mean_execution_time /= n;
        mean_memory_usage /= n;
        mean_priority /= n;

        self.mean_values = vec![mean_execution_time, mean_memory_usage, mean_priority];

        // Normalize features
        let x: Vec<Vec<f64>> = historical_data
            .iter()
            .map(|data| {
                vec![
                    (data.execution_time.as_secs_f64() - mean_execution_time) / mean_execution_time.max(1.0),
                    (data.memory_usage as f64 - mean_memory_usage) / mean_memory_usage.max(1.0),
                    self.normalize_processing_unit(&data.unit_type),
                    (data.priority as f64 - mean_priority) / mean_priority.max(1.0),
                ]
            })
            .collect();

        let y: Vec<f64> = historical_data
            .iter()
            .map(|data| data.execution_time.as_secs_f64() / mean_execution_time.max(1.0))
            .collect();

        // Validate features
        if x.iter().any(|features| features.iter().any(|&f| f.is_nan() || f.is_infinite())) {
            return Err(XpuOptimizerError::MLOptimizationError("Invalid feature values after normalization".to_string()));
        }

        // Adaptive gradient descent
        let mut learning_rate = 0.01;
        let iterations = 1000;
        let m = x.len() as f64;
        let mut prev_error = f64::MAX;

        for iter in 0..iterations {
            let mut gradient = vec![0.0; self.coefficients.len()];
            let mut total_error = 0.0;

            for (xi, &yi) in x.iter().zip(&y) {
                let h: f64 = xi.iter().zip(&self.coefficients).map(|(xi, ci)| xi * ci).sum();
                let error = h - yi;
                total_error += error * error;

                for (j, &xij) in xi.iter().enumerate() {
                    gradient[j] += error * xij / m;
                }
            }

            // Adjust learning rate if error is increasing
            if total_error > prev_error {
                learning_rate *= 0.5;
            }
            prev_error = total_error;

            // Update coefficients with validation
            for (coeff, grad) in self.coefficients.iter_mut().zip(gradient.iter()) {
                let new_coeff = *coeff - learning_rate * grad;
                if new_coeff.is_finite() {
                    *coeff = new_coeff;
                }
            }

            // Early stopping if converged
            if total_error < 1e-6 || learning_rate < 1e-10 {
                log::info!("Training converged after {} iterations", iter);
                break;
            }
        }

        Ok(())
    }

    fn predict(&self, task_data: &HistoricalTaskData) -> Result<TaskPrediction, XpuOptimizerError> {
        // Check if model has valid coefficients (not all zeros)
        if self.coefficients.iter().all(|&x| x == 0.0) {
            return Err(XpuOptimizerError::MLOptimizationError(
                "Model not trained: coefficients are all zero".to_string()
            ));
        }

        let raw_features = [
            task_data.execution_time.as_secs_f64(),
            task_data.memory_usage as f64,
            self.normalize_processing_unit(&task_data.unit_type),
            task_data.priority as f64,
        ];

        // Validate input features
        if raw_features.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(XpuOptimizerError::MLOptimizationError("Invalid input features".to_string()));
        }

        // Normalize features using mean values
        let features: Vec<f64> = raw_features.iter()
            .zip(self.mean_values.iter())
            .map(|(&feat, &mean)| if mean != 0.0 { feat / mean } else { feat })
            .collect();

        // Additional validation after normalization
        if features.iter().any(|&x| x.is_nan() || x.is_infinite() || x.abs() > 1000.0) {
            return Err(XpuOptimizerError::MLOptimizationError("Feature normalization produced invalid values".to_string()));
        }

        let prediction: f64 = self
            .coefficients
            .iter()
            .zip(features.iter())
            .map(|(coef, feat)| coef * feat)
            .sum();

        if prediction.is_nan() || prediction.is_infinite() {
            return Err(XpuOptimizerError::MLOptimizationError("Invalid prediction value".to_string()));
        }

        // Ensure prediction is reasonable (within expected bounds)
        let max_duration = 3600.0; // 1 hour maximum prediction
        let min_duration = 0.1; // 100ms minimum prediction
        let clamped_prediction = prediction.max(min_duration).min(max_duration);

        Ok(TaskPrediction {
            task_id: task_data.task_id,
            estimated_duration: Duration::from_secs_f64(clamped_prediction),
            estimated_resource_usage: (clamped_prediction * 100.0 / max_duration) as usize,
            recommended_processing_unit: task_data.unit_type.clone(),
        })
    }

    fn clone_box(&self) -> Arc<Mutex<dyn MLModel + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }

    fn set_policy(&mut self, policy: &str) -> Result<(), XpuOptimizerError> {
        self.policy = policy.to_string();
        log::info!("Setting ML optimization policy to: {}", policy);
        match policy {
            "default" => {
                // Use default settings
            },
            "aggressive" => {
                // Implement more aggressive learning rate or more iterations
            },
            "conservative" => {
                // Implement more conservative learning rate or fewer iterations
            },
            _ => return Err(XpuOptimizerError::MLOptimizationError(format!("Unknown policy: {}", policy))),
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct DefaultMLOptimizer {
    ml_model: Arc<Mutex<dyn MLModel + Send + Sync>>,
    policy: String,
}

impl DefaultMLOptimizer {
    pub fn new(ml_model: Option<Arc<Mutex<dyn MLModel + Send + Sync>>>) -> Self {
        DefaultMLOptimizer {
            ml_model: ml_model.unwrap_or_else(|| Arc::new(Mutex::new(SimpleRegressionModel::new()))),
            policy: "default".to_string(),
        }
    }
}

impl MachineLearningOptimizer for DefaultMLOptimizer {
    fn optimize(
        &self,
        historical_data: &[TaskExecutionData],
        _processing_units: &[Arc<Mutex<dyn ProcessingUnitTrait + Send + Sync>>],
    ) -> Result<Scheduler, XpuOptimizerError> {
        let mut model = self.ml_model.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?;

        model.train(historical_data)
            .map_err(|e| XpuOptimizerError::MLOptimizationError(format!("Failed to train model: {}", e)))?;

        log::info!("ML model trained successfully with {} historical data points", historical_data.len());

        // Here we could use processing_units to make more informed decisions
        // For now, we're just creating a new AIOptimizedScheduler
        let scheduler = Scheduler::AIOptimized(AIOptimizedScheduler::new(Arc::clone(&self.ml_model)));

        log::info!("Created new AIOptimizedScheduler based on trained model");

        Ok(scheduler)
    }

    fn clone_box(&self) -> Arc<Mutex<dyn MachineLearningOptimizer + Send + Sync>> {
        Arc::new(Mutex::new(self.clone()))
    }

    fn generate_token(&self) -> Result<String, XpuOptimizerError> {
        Ok(uuid::Uuid::new_v4().to_string())
    }

    fn set_policy(&mut self, policy: &str) -> Result<(), XpuOptimizerError> {
        self.policy = policy.to_string();
        log::info!("Setting DefaultMLOptimizer policy to: {}", policy);
        let mut model = self.ml_model.lock()
            .map_err(|e| XpuOptimizerError::LockError(format!("Failed to lock ML model: {}", e)))?;
        model.set_policy(policy)
    }
}

use crate::task_scheduling::Task;
use crate::XpuOptimizerError;

#[derive(Debug, Clone, Copy)]
pub enum CloudOffloadingPolicy {
    Default,
    Always,
    Never,
}

pub trait CloudOffloader: Send + Sync {
    fn offload_task(&self, task: &Task) -> Result<(), XpuOptimizerError>;
    fn set_policy(&mut self, policy: CloudOffloadingPolicy);
}

pub struct DefaultCloudOffloader {
    policy: CloudOffloadingPolicy,
}

impl DefaultCloudOffloader {
    pub fn new() -> Self {
        DefaultCloudOffloader {
            policy: CloudOffloadingPolicy::Default,
        }
    }

    pub fn set_policy(&mut self, policy: CloudOffloadingPolicy) {
        self.policy = policy;
        log::info!("Cloud offloading policy set to {:?}", policy);
    }
}

// Removed duplicate implementation of set_policy for CloudOffloader trait

impl CloudOffloader for DefaultCloudOffloader {
    fn set_policy(&mut self, policy: CloudOffloadingPolicy) {
        self.policy = policy;
        log::info!("Cloud offloading policy set to {:?}", policy);
    }

    fn offload_task(&self, task: &Task) -> Result<(), XpuOptimizerError> {
        match self.policy {
            CloudOffloadingPolicy::Never => {
                log::info!("Task {} not offloaded due to Never policy", task.id);
                Ok(())
            },
            CloudOffloadingPolicy::Always => {
                log::info!("Offloading task {} to cloud (Always policy)", task.id);
                self.perform_offload(task)
            },
            CloudOffloadingPolicy::Default => {
                if task.id % 5 == 0 {
                    log::info!("Offloading task {} to cloud (Default policy)", task.id);
                    self.perform_offload(task)
                } else {
                    log::info!("Task {} not offloaded (Default policy)", task.id);
                    Ok(())
                }
            },
        }
    }
}

impl DefaultCloudOffloader {
    fn perform_offload(&self, task: &Task) -> Result<(), XpuOptimizerError> {
        // Simulate potential network errors or other cloud-related issues
        if task.id % 7 == 0 {
            log::error!("Failed to offload task {} due to simulated network error", task.id);
            Err(XpuOptimizerError::CloudOffloadingError(format!("Failed to offload task {}", task.id)))
        } else {
            log::info!("Successfully offloaded task {} to cloud", task.id);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use crate::task_scheduling::{Task, ProcessingUnitType};

    #[test]
    fn test_default_cloud_offloader() {
        let offloader = DefaultCloudOffloader::new();
        let task = Task {
            id: 1,
            priority: 1,
            dependencies: vec![],
            execution_time: Duration::from_secs(1),
            memory_requirement: 100,
            secure: false,
            estimated_duration: Duration::from_secs(1),
            estimated_resource_usage: 100,
            unit_type: ProcessingUnitType::CPU,
        };

        assert!(offloader.offload_task(&task).is_ok());
    }
}

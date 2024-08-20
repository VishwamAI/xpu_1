use crate::task_scheduling::{ProcessingUnitType, Task, ProcessingUnitTrait};
use crate::power_management::{PowerState, EnergyProfile, PowerManagementError};
use crate::XpuOptimizerError;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct TPU {
    id: usize,
    unit_type: ProcessingUnitType,
    current_load: Duration,
    processing_power: f64,
    power_state: PowerState,
    energy_profile: EnergyProfile,
}

impl TPU {
    pub fn new(id: usize, processing_power: f64) -> Self {
        TPU {
            id,
            unit_type: ProcessingUnitType::TPU,
            current_load: Duration::new(0, 0),
            processing_power,
            power_state: PowerState::Normal,
            energy_profile: EnergyProfile::default(),
        }
    }

    fn calculate_execution_time(&self, task: &Task) -> Duration {
        let base_time = task.execution_time;
        base_time.mul_f64(1.0 / self.processing_power)
    }

    fn adjust_power_state(&mut self) {
        self.power_state = match self.current_load.as_secs_f64() / self.processing_power {
            load if load < 0.3 => PowerState::LowPower,
            load if load < 0.7 => PowerState::Normal,
            _ => PowerState::HighPerformance,
        };
    }
}

impl ProcessingUnitTrait for TPU {
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
        Ok(self.current_load.as_secs_f64() / self.processing_power)
    }

    fn can_handle_task(&self, task: &Task) -> Result<bool, XpuOptimizerError> {
        Ok(task.unit_type == ProcessingUnitType::TPU &&
           self.current_load + task.execution_time <= Duration::from_secs_f64(self.processing_power))
    }

    fn assign_task(&mut self, task: &Task) -> Result<(), XpuOptimizerError> {
        if self.can_handle_task(task)? {
            self.current_load = self.current_load.saturating_add(task.execution_time);
            Ok(())
        } else {
            Err(XpuOptimizerError::ResourceAllocationError(
                format!("TPU cannot handle task {}", task.id)
            ))
        }
    }

    fn set_power_state(&mut self, state: PowerState) -> Result<(), PowerManagementError> {
        self.power_state = state;
        match self.power_state {
            PowerState::LowPower => self.energy_profile.consumption_rate *= 0.7,
            PowerState::Normal => self.energy_profile.consumption_rate = 1.0,
            PowerState::HighPerformance => self.energy_profile.consumption_rate *= 1.3,
        }
        Ok(())
    }

    fn get_available_capacity(&self) -> Result<Duration, XpuOptimizerError> {
        Ok(Duration::from_secs_f64(self.processing_power) - self.current_load)
    }

    fn clone_box(&self) -> Box<dyn ProcessingUnitTrait + Send + Sync> {
        Box::new(self.clone())
    }

    fn execute_task(&mut self, task: &Task) -> Result<Duration, XpuOptimizerError> {
        let execution_time = self.calculate_execution_time(task);
        self.assign_task(task)?;
        self.adjust_power_state();

        log::info!("TPU processed task {} in {:?}", task.id, execution_time);
        Ok(execution_time)
    }

    fn set_energy_profile(&mut self, profile: EnergyProfile) -> Result<(), PowerManagementError> {
        self.energy_profile = profile;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

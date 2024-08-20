use crate::task_scheduling::{ProcessingUnit, ProcessingUnitType, Task, ProcessingUnitTrait};
use crate::power_management::{PowerState, EnergyProfile, PowerManagementError};
use crate::XpuOptimizerError;
use std::time::Duration;

#[derive(Debug)]
pub struct CPU {
    processing_unit: ProcessingUnit,
}

impl CPU {
    pub fn new(id: usize, processing_power: f64) -> Self {
        CPU {
            processing_unit: ProcessingUnit {
                id,
                unit_type: ProcessingUnitType::CPU,
                current_load: Duration::new(0, 0),
                processing_power,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            },
        }
    }

    fn adjust_power_consumption(&mut self) {
        // Simulate power consumption adjustment
        match self.processing_unit.power_state {
            PowerState::LowPower => self.processing_unit.energy_profile.consumption_rate = 0.5,
            PowerState::Normal => self.processing_unit.energy_profile.consumption_rate = 1.0,
            PowerState::HighPerformance => self.processing_unit.energy_profile.consumption_rate = 2.0,
        }
    }
}

impl ProcessingUnitTrait for CPU {
    fn get_id(&self) -> usize {
        self.processing_unit.id
    }

    fn get_unit_type(&self) -> Result<ProcessingUnitType, XpuOptimizerError> {
        Ok(self.processing_unit.unit_type.clone())
    }

    fn get_current_load(&self) -> Result<Duration, XpuOptimizerError> {
        Ok(self.processing_unit.current_load)
    }

    fn get_processing_power(&self) -> Result<f64, XpuOptimizerError> {
        Ok(self.processing_unit.processing_power)
    }

    fn get_power_state(&self) -> Result<PowerState, XpuOptimizerError> {
        Ok(self.processing_unit.power_state.clone())
    }

    fn get_energy_profile(&self) -> Result<&EnergyProfile, XpuOptimizerError> {
        Ok(&self.processing_unit.energy_profile)
    }

    fn get_load_percentage(&self) -> Result<f64, XpuOptimizerError> {
        Ok(self.processing_unit.current_load.as_secs_f64() / self.processing_unit.processing_power)
    }

    fn can_handle_task(&self, task: &Task) -> Result<bool, XpuOptimizerError> {
        Ok(task.unit_type == ProcessingUnitType::CPU &&
           self.processing_unit.current_load + task.execution_time <= Duration::from_secs_f64(self.processing_unit.processing_power))
    }

    fn assign_task(&mut self, task: &Task) -> Result<(), XpuOptimizerError> {
        if self.can_handle_task(task)? {
            self.processing_unit.current_load += task.execution_time;
            Ok(())
        } else {
            Err(XpuOptimizerError::ResourceAllocationError(
                format!("CPU cannot handle task {}", task.id)
            ))
        }
    }

    fn set_power_state(&mut self, state: PowerState) -> Result<(), PowerManagementError> {
        self.processing_unit.power_state = state;
        self.adjust_power_consumption();
        Ok(())
    }

    fn get_available_capacity(&self) -> Result<Duration, XpuOptimizerError> {
        Ok(Duration::from_secs_f64(self.processing_unit.processing_power) - self.processing_unit.current_load)
    }

    fn clone_box(&self) -> Box<dyn ProcessingUnitTrait + Send + Sync> {
        Box::new(self.clone())
    }

    fn execute_task(&mut self, task: &Task) -> Result<Duration, XpuOptimizerError> {
        self.assign_task(task)?;
        Ok(task.execution_time)
    }

    fn set_energy_profile(&mut self, profile: EnergyProfile) -> Result<(), PowerManagementError> {
        self.processing_unit.energy_profile = profile;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Clone for CPU {
    fn clone(&self) -> Self {
        CPU {
            processing_unit: self.processing_unit.clone(),
        }
    }
}

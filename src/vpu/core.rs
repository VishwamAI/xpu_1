use crate::task_scheduling::{ProcessingUnit, ProcessingUnitType, Task, ProcessingUnitTrait};
use crate::power_management::{PowerState, EnergyProfile, PowerManagementError};
use crate::XpuOptimizerError;
use std::time::Duration;

#[derive(Debug)]
pub struct VPU {
    processing_unit: ProcessingUnit,
}

impl VPU {
    pub fn new(id: usize, processing_power: f64) -> Self {
        VPU {
            processing_unit: ProcessingUnit {
                id,
                unit_type: ProcessingUnitType::VPU,
                current_load: Duration::new(0, 0),
                processing_power,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            },
        }
    }

    fn adjust_power_consumption(&mut self) {
        match self.processing_unit.power_state {
            PowerState::LowPower => self.processing_unit.energy_profile.consumption_rate *= 0.7,
            PowerState::Normal => self.processing_unit.energy_profile.consumption_rate = 1.0,
            PowerState::HighPerformance => self.processing_unit.energy_profile.consumption_rate *= 1.3,
        }
    }

    pub fn optimize_for_visual_processing(&mut self) {
        // Implement VPU-specific optimizations for visual processing tasks
        self.processing_unit.processing_power *= 1.2; // Increase processing power for visual tasks
    }
}

impl ProcessingUnitTrait for VPU {
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
        Ok(task.unit_type == ProcessingUnitType::VPU &&
           self.processing_unit.current_load + task.execution_time <= Duration::from_secs_f64(self.processing_unit.processing_power))
    }

    fn assign_task(&mut self, task: &Task) -> Result<(), XpuOptimizerError> {
        if self.can_handle_task(task)? {
            self.processing_unit.current_load = self.processing_unit.current_load.saturating_add(task.execution_time);
            Ok(())
        } else {
            Err(XpuOptimizerError::ResourceAllocationError(
                format!("VPU cannot handle task {}", task.id)
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
        if !self.can_handle_task(task)? {
            return Err(XpuOptimizerError::TaskExecutionError(format!("VPU cannot handle task {}", task.id)));
        }

        let processing_time = task.execution_time.div_f64(self.processing_unit.processing_power);
        self.assign_task(task)?;

        log::info!("VPU processed task {} in {:?}", task.id, processing_time);
        Ok(processing_time)
    }

    fn set_energy_profile(&mut self, profile: EnergyProfile) -> Result<(), PowerManagementError> {
        self.processing_unit.energy_profile = profile;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Clone for VPU {
    fn clone(&self) -> Self {
        VPU {
            processing_unit: self.processing_unit.clone(),
        }
    }
}
